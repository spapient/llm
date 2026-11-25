//! `.k` 模板 → `kernel::Program`：配置条件求值、C 子集词法 / 表达式解析与原子绑定。

use crate::ast::{Token, lex_c, preprocess};
use crate::cap::AtomTable;
use crate::decl::{OpDecl, TileCombo};
use crate::kernel::{
    Access, AtomArg, AtomCall, CompareKind, Expr, Function, LocalDecl, LocalType, LogicalKind,
    Parameter, Program, ScalarType, Statement, TypeAlias, ValueType,
};
use crate::substitute::error;
use prof::atom::Prim;

/// 把一份声明、tile 与已协商原子表 lower 成完整程序 IR。
///
/// 所有模板必须成功解析；没有文本回退路径。配置 `#if` 在构型确定后求值，不能成立的
/// `#error` 或其他未知指令都会报错。
pub fn parse<'a>(
    op: &OpDecl,
    combo: &TileCombo,
    atoms: &AtomTable<'a>,
) -> diagnostic::Result<Program<'a>> {
    let resolved = crate::ast::resolve_slots(op, combo, &op.template, 1)?;
    let source = preprocess(&resolved, const_value)?;
    let tokens = lex_c(&source)?;
    let program_atoms: AtomTable<'a> = atoms.clone();
    Parser::new(tokens, Some(atoms)).program(op, program_atoms)
}
fn const_value(text: &str) -> diagnostic::Result<i64> {
    let mut parser = Parser::new(lex_c(text)?, None);
    let expression = parser.expression(0)?;
    parser.finish()?;
    crate::expr::const_int(&expression)
        .and_then(|value| i64::try_from(value).ok())
        .ok_or_else(|| error(1, format!("`{text}` 不是可求值的整数常量")))
}
#[derive(Clone, Copy)]
enum Binary {
    Arith(crate::kernel::Arith),
    Compare(CompareKind),
    Logical(LogicalKind),
}
struct Parser<'s, 'table, 'a> {
    tokens: Vec<Token<'s>>,
    at: usize,
    atoms: Option<&'table AtomTable<'a>>,
}

impl<'s, 'table, 'a> Parser<'s, 'table, 'a> {
    fn new(tokens: Vec<Token<'s>>, atoms: Option<&'table AtomTable<'a>>) -> Self {
        Self {
            tokens,
            at: 0,
            atoms,
        }
    }

    fn current(&self) -> Token<'s> {
        self.tokens.get(self.at).copied().unwrap_or(Token {
            text: "<eof>",
            line: self.tokens.last().map_or(1, |token| token.line),
        })
    }

    fn at_text(&self, text: &str) -> bool {
        self.current().text == text
    }

    fn consume(&mut self, text: &str) -> bool {
        if !self.at_text(text) {
            return false;
        }
        self.at += 1;
        true
    }

    fn require(&mut self, text: &str) -> diagnostic::Result<Token<'s>> {
        if !self.at_text(text) {
            return Err(error(
                self.current().line,
                format!("期望 `{text}`，得到 `{}`", self.current().text),
            ));
        }
        let token = self.current();
        self.at += 1;
        Ok(token)
    }

    fn finish(&self) -> diagnostic::Result<()> {
        if self.at == self.tokens.len() {
            return Ok(());
        }
        Err(error(
            self.current().line,
            format!("多余 token `{}`", self.current().text),
        ))
    }

    fn program(&mut self, op: &OpDecl, atoms: AtomTable<'a>) -> diagnostic::Result<Program<'a>> {
        let expected = format!("cutile_{}_kernel", op.name);
        let (aliases, helpers, main) = self.functions(&expected)?;
        let main = main.ok_or_else(|| error(1, format!("模板缺少 `{expected}`")))?;
        Ok(Program::new(op.name.clone(), main.params, atoms, main.body)
            .with_aliases(aliases)
            .with_helpers(helpers))
    }

    fn functions(
        &mut self,
        expected: &str,
    ) -> diagnostic::Result<(Vec<TypeAlias>, Vec<Function>, Option<Function>)> {
        let (mut aliases, mut helpers, mut main) = (Vec::new(), Vec::new(), None);
        while self.current().text != "<eof>" {
            self.function_step(expected, &mut aliases, &mut helpers, &mut main)?;
        }
        Ok((aliases, helpers, main))
    }

    fn function_step(
        &mut self,
        expected: &str,
        aliases: &mut Vec<TypeAlias>,
        helpers: &mut Vec<Function>,
        main: &mut Option<Function>,
    ) -> diagnostic::Result<()> {
        if self.consume("typedef") {
            aliases.push(self.type_alias()?);
            return Ok(());
        }
        let function = self.function()?;
        register_function(function, expected, helpers, main)
    }

    fn type_alias(&mut self) -> diagnostic::Result<TypeAlias> {
        let ty = self
            .scalar_type()
            .ok_or_else(|| error(self.current().line, "typedef 缺目标类型"))?;
        let name = self.expect_identifier()?.to_string();
        self.require(";")?;
        Ok(TypeAlias { name, ty })
    }

    fn function(&mut self) -> diagnostic::Result<Function> {
        let (result, name, params) = self.signature()?;
        self.require("{")?;
        let body = self.block()?;
        Ok(Function {
            name,
            result,
            params,
            body,
        })
    }

    fn signature(&mut self) -> diagnostic::Result<(ScalarType, String, Vec<Parameter>)> {
        self.consume("static");
        self.consume("inline");
        let result = self
            .scalar_type()
            .ok_or_else(|| error(self.current().line, "函数缺返回类型"))?;
        let name = self.expect_identifier()?.to_string();
        let params = self.parameters()?;
        Ok((result, name, params))
    }

    fn parameters(&mut self) -> diagnostic::Result<Vec<Parameter>> {
        self.require("(")?;
        let mut params = Vec::new();
        while self.parameter_step(&mut params)? {}
        self.require(")")?;
        Ok(params)
    }

    fn parameter_step(&mut self, params: &mut Vec<Parameter>) -> diagnostic::Result<bool> {
        params.push(self.parameter()?);
        Ok(self.consume(","))
    }

    fn parameter(&mut self) -> diagnostic::Result<Parameter> {
        let constant = self.consume("const");
        let element = self
            .scalar_type()
            .ok_or_else(|| error(self.current().line, "未知形参类型"))?;
        let pointers = self.pointer_count();
        let name = self.expect_identifier()?.to_string();
        let ty = parameter_type(element, pointers, constant);
        Ok(Parameter { name, ty })
    }

    fn pointer_count(&mut self) -> u8 {
        let mut pointers = 0;
        while self.consume("*") {
            pointers += 1;
        }
        pointers
    }

    fn block(&mut self) -> diagnostic::Result<Vec<Statement>> {
        let mut body = Vec::new();
        while !self.consume("}") {
            if self.current().text == "<eof>" {
                return Err(error(self.current().line, "语句块缺少 `}`"));
            }
            body.extend(self.statement()?);
        }
        Ok(body)
    }

    fn statement(&mut self) -> diagnostic::Result<Vec<Statement>> {
        if let Some(statements) = self.control_statement()? {
            return Ok(statements);
        }
        if self.is_declaration() {
            return self.declaration(true);
        }
        self.simple_statement().map(|node| vec![node])
    }

    fn control_statement(&mut self) -> diagnostic::Result<Option<Vec<Statement>>> {
        if self.consume("{") {
            return self.block().map(|body| Some(vec![Statement::Block(body)]));
        }
        if self.consume("for") {
            return self.for_loop().map(|node| Some(vec![node]));
        }
        if self.consume("if") {
            return self.if_branch().map(|node| Some(vec![node]));
        }
        self.special_statement()
    }

    fn special_statement(&mut self) -> diagnostic::Result<Option<Vec<Statement>>> {
        if self.consume("@") {
            return self.atom_statement().map(|node| Some(vec![node]));
        }
        if self.at_text("#") {
            return self.pragma().map(|node| Some(vec![node]));
        }
        Ok(None)
    }

    fn pragma(&mut self) -> diagnostic::Result<Statement> {
        let line = self.current().line;
        let name = self.pragma_name()?;
        let value = self.pragma_value(line)?;
        Ok(Statement::Pragma { name, value })
    }

    fn pragma_name(&mut self) -> diagnostic::Result<String> {
        self.require("#")?;
        self.require("pragma")?;
        self.expect_identifier().map(str::to_string)
    }

    fn pragma_value(&mut self, line: usize) -> diagnostic::Result<Option<u32>> {
        let mut value = None;
        if self.current().line == line {
            value = Some(
                self.current()
                    .text
                    .parse::<u32>()
                    .map_err(|_| error(line, "pragma 参数必须是正整数"))?,
            );
            self.at += 1;
        }
        if self.current().line == line || value == Some(0) {
            return Err(error(line, "pragma 只能带一个正整数参数"));
        }
        Ok(value)
    }

    fn simple_statement(&mut self) -> diagnostic::Result<Statement> {
        if let Some(statement) = self.keyword_statement()? {
            return Ok(statement);
        }
        if let Some(statement) = self.return_statement()? {
            return Ok(statement);
        }
        self.expression_statement()
    }

    fn keyword_statement(&mut self) -> diagnostic::Result<Option<Statement>> {
        for keyword in ["continue", "break"] {
            if self.consume(keyword) {
                self.require(";")?;
                return Ok(Some(if keyword == "continue" {
                    Statement::Continue
                } else {
                    Statement::Break
                }));
            }
        }
        Ok(None)
    }

    fn return_statement(&mut self) -> diagnostic::Result<Option<Statement>> {
        if !self.consume("return") {
            return Ok(None);
        }
        let value = (!self.at_text(";"))
            .then(|| self.expression(0))
            .transpose()?;
        self.require(";")?;
        Ok(Some(Statement::Return(value)))
    }

    fn expression_statement(&mut self) -> diagnostic::Result<Statement> {
        let value = self.expression(0)?;
        self.require(";")?;
        Ok(Statement::Expr(value))
    }

    fn body(&mut self) -> diagnostic::Result<Vec<Statement>> {
        if self.consume("{") {
            return self.block();
        }
        self.statement()
    }

    fn if_branch(&mut self) -> diagnostic::Result<Statement> {
        let condition = self.condition()?;
        let then_branch = self.body()?;
        let else_branch = self.else_branch()?;
        Ok(Statement::If {
            condition,
            then_branch,
            else_branch,
        })
    }

    fn condition(&mut self) -> diagnostic::Result<Expr> {
        self.require("(")?;
        let condition = self.expression(0)?;
        self.require(")")?;
        Ok(condition)
    }

    fn else_branch(&mut self) -> diagnostic::Result<Vec<Statement>> {
        if self.consume("else") {
            self.body()
        } else {
            Ok(Vec::new())
        }
    }

    fn for_loop(&mut self) -> diagnostic::Result<Statement> {
        let (init, condition, step) = self.for_header()?;
        Ok(Statement::For {
            init: init.map(Box::new),
            condition,
            step,
            body: self.body()?,
        })
    }

    fn for_header(
        &mut self,
    ) -> diagnostic::Result<(Option<Statement>, Option<Expr>, Option<Expr>)> {
        self.require("(")?;
        let init = self.for_init()?;
        let condition = self.delimited_expression(";")?;
        let step = self.delimited_expression(")")?;
        Ok((init, condition, step))
    }

    fn delimited_expression(&mut self, delimiter: &str) -> diagnostic::Result<Option<Expr>> {
        let value = self.optional_expression(delimiter)?;
        self.require(delimiter)?;
        Ok(value)
    }

    fn for_init(&mut self) -> diagnostic::Result<Option<Statement>> {
        if self.consume(";") {
            return Ok(None);
        }
        if self.is_declaration() {
            return self.for_declaration();
        }
        let value = Statement::Expr(self.expression(0)?);
        self.require(";")?;
        Ok(Some(value))
    }

    fn for_declaration(&mut self) -> diagnostic::Result<Option<Statement>> {
        let declarations = self.declaration(true)?;
        if declarations.len() == 1 {
            return Ok(declarations.into_iter().next());
        }
        Err(error(self.current().line, "for 初始化只支持一条声明"))
    }

    fn optional_expression(&mut self, delimiter: &str) -> diagnostic::Result<Option<Expr>> {
        if self.at_text(delimiter) {
            return Ok(None);
        }
        self.expression(0).map(Some)
    }

    fn is_declaration(&self) -> bool {
        [
            "const", "float", "float4", "double", "int", "signed", "unsigned", "long", "size_t",
            "u8", "half", "__half", "bf16", "bfloat",
        ]
        .contains(&self.current().text)
    }
    fn scalar_type(&mut self) -> Option<ScalarType> {
        const TYPES: &[(&str, ScalarType)] = &[
            ("float", ScalarType::F32),
            ("float4", ScalarType::F32x4),
            ("half", ScalarType::F16),
            ("__half", ScalarType::F16),
            ("bf16", ScalarType::Bf16),
            ("bfloat", ScalarType::Bf16),
            ("double", ScalarType::F64),
            ("void", ScalarType::Void),
            ("long", ScalarType::I64),
            ("size_t", ScalarType::Usize),
            ("u8", ScalarType::U8),
        ];
        let found = TYPES.iter().find(|(name, _)| *name == self.current().text);
        if let Some((_, ty)) = found {
            self.at += 1;
            return Some(*ty);
        }
        self.qualified_integer()
    }

    fn qualified_integer(&mut self) -> Option<ScalarType> {
        let signed = self.consume("signed");
        let unsigned = !signed && self.consume("unsigned");
        if self.consume("char") {
            return Some(if unsigned {
                ScalarType::U8
            } else {
                ScalarType::I8
            });
        }
        self.qualified_int(signed, unsigned)
    }

    fn qualified_int(&mut self, signed: bool, unsigned: bool) -> Option<ScalarType> {
        if !(signed || unsigned) && !self.at_text("int") {
            return None;
        }
        self.consume("int");
        Some(if unsigned {
            ScalarType::U32
        } else {
            ScalarType::I32
        })
    }

    fn declaration(&mut self, semicolon: bool) -> diagnostic::Result<Vec<Statement>> {
        let constant = self.consume("const");
        let element = self
            .scalar_type()
            .ok_or_else(|| error(self.current().line, "未知局部类型"))?;
        let pointers = self.pointer_count();
        let declarations = self.declarators(element, pointers, constant)?;
        self.finish_declaration(declarations, semicolon)
    }

    fn declarators(
        &mut self,
        element: ScalarType,
        pointers: u8,
        constant: bool,
    ) -> diagnostic::Result<Vec<Statement>> {
        let mut declarations = Vec::new();
        loop {
            declarations.push(self.local_decl(element, pointers, constant)?);
            if !self.consume(",") {
                break;
            }
        }
        Ok(declarations)
    }

    fn local_decl(
        &mut self,
        element: ScalarType,
        pointers: u8,
        constant: bool,
    ) -> diagnostic::Result<Statement> {
        let name = self.expect_identifier()?.to_string();
        let array = self.array_len()?;
        let init = self.initializer()?;
        Ok(Statement::Decl(LocalDecl {
            name,
            ty: LocalType {
                element,
                pointers,
                constant,
                array,
            },
            init,
        }))
    }

    fn array_len(&mut self) -> diagnostic::Result<Option<usize>> {
        if !self.consume("[") {
            return Ok(None);
        }
        let size = self.expression(0)?;
        self.finish_array(size)
    }

    fn finish_array(&mut self, size: Expr) -> diagnostic::Result<Option<usize>> {
        self.require("]")?;
        let size = crate::expr::const_int(&size)
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| error(self.current().line, "数组长度必须是编译期非负整数"))?;
        Ok(Some(size))
    }

    fn initializer(&mut self) -> diagnostic::Result<Option<Expr>> {
        if self.consume("=") {
            self.expression(0).map(Some)
        } else {
            Ok(None)
        }
    }

    fn finish_declaration(
        &mut self,
        declarations: Vec<Statement>,
        semicolon: bool,
    ) -> diagnostic::Result<Vec<Statement>> {
        if semicolon {
            self.require(";")?;
        }
        Ok(declarations)
    }

    fn atom_statement(&mut self) -> diagnostic::Result<Statement> {
        let line = self.current().line;
        let prim = self.prim(line)?;
        let args = self.atom_call()?;
        let atom = self.resolve_atom(prim, &args)?;
        Ok(Statement::Atom(AtomCall { atom, args, line }))
    }

    fn prim(&mut self, line: usize) -> diagnostic::Result<Prim> {
        Prim::parse(self.expect_identifier()?).ok_or_else(|| error(line, "模板调用了未知原语"))
    }

    fn atom_call(&mut self) -> diagnostic::Result<Vec<AtomArg>> {
        self.require("(")?;
        let args = self.atom_args()?;
        self.require(")")?;
        self.require(";")?;
        Ok(args)
    }

    fn atom_args(&mut self) -> diagnostic::Result<Vec<AtomArg>> {
        let mut args = Vec::new();
        while !self.at_text(")") {
            args.push(self.atom_arg()?);
            if !self.consume(",") {
                break;
            }
        }
        Ok(args)
    }

    fn resolve_atom(
        &mut self,
        prim: Prim,
        args: &[AtomArg],
    ) -> diagnostic::Result<crate::cap::AtomId> {
        let named: Vec<(&str, &str)> = args
            .iter()
            .filter_map(|arg| match arg {
                AtomArg::Named { key, value } => Some((key.as_str(), value.as_str())),
                AtomArg::Expr(_) => None,
            })
            .collect();
        self.atoms
            .and_then(|atoms| atoms.resolve(prim, &named))
            .ok_or_else(|| error(1, format!("原语 `{prim}` 没有匹配的已协商原子")))
    }

    fn atom_arg(&mut self) -> diagnostic::Result<AtomArg> {
        let named = self
            .tokens
            .get(self.at + 1)
            .is_some_and(|token| token.text == ":");
        if named {
            let key = self.expect_identifier()?.to_string();
            self.require(":")?;
            let value = self.current().text.to_string();
            if value == "<eof>" {
                return Err(error(self.current().line, "原子命名参数缺值"));
            }
            self.at += 1;
            return Ok(AtomArg::Named { key, value });
        }
        self.expression(0).map(AtomArg::Expr)
    }

    fn expression(&mut self, minimum: u8) -> diagnostic::Result<Expr> {
        let left = self.prefix()?;
        self.expression_tail(left, minimum)
    }

    fn expression_tail(&mut self, mut left: Expr, minimum: u8) -> diagnostic::Result<Expr> {
        loop {
            match self.advance_expression(left, minimum)? {
                Step::Next(next) => left = next,
                Step::Done(done) => return Ok(done),
            }
        }
    }

    fn advance_expression(&mut self, left: Expr, minimum: u8) -> diagnostic::Result<Step> {
        let left = self.postfix_all(left)?;
        self.selection_or_binary(left, minimum)
    }

    fn selection_or_binary(&mut self, left: Expr, minimum: u8) -> diagnostic::Result<Step> {
        if !self.at_text("?") || minimum > 2 {
            return self.assign_or_binary(left, minimum);
        }
        self.at += 1;
        self.selection_node(left).map(Step::Next)
    }

    fn selection_node(&mut self, left: Expr) -> diagnostic::Result<Expr> {
        let when_true = self.expression(0)?;
        self.require(":")?;
        let when_false = self.expression(2)?;
        Ok(Expr::Select {
            condition: Box::new(left),
            when_true: Box::new(when_true),
            when_false: Box::new(when_false),
        })
    }

    fn assign_or_binary(&mut self, left: Expr, minimum: u8) -> diagnostic::Result<Step> {
        let Some(op) = self.assignment() else {
            return self.binary_step(left, minimum);
        };
        if minimum > 1 {
            return self.binary_step(left, minimum);
        }
        self.at += 1;
        let value = self.expression(1)?;
        Ok(Step::Next(Expr::Assign {
            target: Box::new(left),
            op,
            value: Box::new(value),
        }))
    }

    fn binary_step(&mut self, left: Expr, minimum: u8) -> diagnostic::Result<Step> {
        let Some((left_bp, right_bp, op)) = self.binary() else {
            return Ok(Step::Done(left));
        };
        if left_bp < minimum {
            return Ok(Step::Done(left));
        }
        self.at += 1;
        let right = self.expression(right_bp)?;
        Ok(Step::Next(binary_node(op, left, right)))
    }

    fn postfix_all(&mut self, mut left: Expr) -> diagnostic::Result<Expr> {
        while self.postfix_start() {
            left = self.postfix(left)?;
        }
        Ok(left)
    }

    fn postfix_start(&self) -> bool {
        self.at_text("(") || self.at_text("[") || self.at_text("++") || self.at_text("--")
    }

    fn prefix(&mut self) -> diagnostic::Result<Expr> {
        if self.consume("(") {
            return self.paren_or_cast();
        }
        self.unary_prefix()
    }

    fn paren_or_cast(&mut self) -> diagnostic::Result<Expr> {
        let saved = self.at;
        if let Some(ty) = self.scalar_type()
            && self.consume(")")
        {
            return self.cast_value(ty);
        }
        self.group(saved)
    }

    fn cast_value(&mut self, ty: ScalarType) -> diagnostic::Result<Expr> {
        let value = self.expression(12)?;
        Ok(Expr::Cast {
            ty,
            value: Box::new(value),
        })
    }

    fn group(&mut self, saved: usize) -> diagnostic::Result<Expr> {
        self.at = saved;
        let value = self.expression(0)?;
        self.require(")")?;
        Ok(value)
    }

    fn unary_prefix(&mut self) -> diagnostic::Result<Expr> {
        for operator in ["++", "--", "+", "-", "!", "~", "&", "*"] {
            if self.consume(operator) {
                let child = Box::new(self.expression(12)?);
                return Ok(crate::expr::prefix_node(operator, child));
            }
        }
        self.leaf()
    }

    fn leaf(&mut self) -> diagnostic::Result<Expr> {
        let token = self.current();
        if token.text == "<eof>" {
            return Err(error(token.line, "表达式意外结束"));
        }
        self.at += 1;
        if crate::expr::numeric_start(token.text) {
            return crate::expr::parse_number(token.text, token.line);
        }
        if crate::expr::identifier_start(token.text) {
            return Ok(Expr::Scalar(token.text.to_string()));
        }
        Err(error(
            token.line,
            format!("`{}` 不能作为表达式起点", token.text),
        ))
    }

    fn postfix(&mut self, left: Expr) -> diagnostic::Result<Expr> {
        if self.at_text("[") {
            return self.subscript(left);
        }
        if self.at_text("(") {
            return self.call(left);
        }
        self.update(left)
    }

    fn subscript(&mut self, left: Expr) -> diagnostic::Result<Expr> {
        self.consume("[");
        let idx = self.expression(0)?;
        self.require("]")?;
        Ok(Expr::Index {
            base: Box::new(left),
            idx: Box::new(idx),
        })
    }

    fn call(&mut self, left: Expr) -> diagnostic::Result<Expr> {
        self.consume("(");
        let Expr::Scalar(function) = left else {
            return Err(error(self.current().line, "只支持直接函数调用"));
        };
        let args = self.call_args()?;
        self.require(")")?;
        Ok(Expr::Call { function, args })
    }

    fn call_args(&mut self) -> diagnostic::Result<Vec<Expr>> {
        let mut args = Vec::new();
        while !self.at_text(")") {
            args.push(self.expression(0)?);
            if !self.consume(",") {
                break;
            }
        }
        Ok(args)
    }

    fn update(&mut self, left: Expr) -> diagnostic::Result<Expr> {
        let delta = if self.consume("++") {
            1
        } else {
            self.require("--").map(|_| -1)?
        };
        Ok(Expr::Update {
            target: Box::new(left),
            delta,
            prefix: false,
        })
    }

    fn assignment(&self) -> Option<Option<crate::kernel::Arith>> {
        use crate::kernel::Arith;
        const RULES: &[(&str, Option<Arith>)] = &[
            ("=", None),
            ("+=", Some(Arith::Add)),
            ("-=", Some(Arith::Sub)),
            ("*=", Some(Arith::Mul)),
            ("/=", Some(Arith::Div)),
            ("%=", Some(Arith::Rem)),
        ];
        RULES
            .iter()
            .find(|(name, _)| *name == self.current().text)
            .map(|(_, op)| *op)
    }

    fn binary(&self) -> Option<(u8, u8, Binary)> {
        use crate::kernel::Arith;
        const RULES: &[(&str, u8, Binary)] = &[
            ("||", 3, Binary::Logical(LogicalKind::Or)),
            ("&&", 4, Binary::Logical(LogicalKind::And)),
            ("|", 5, Binary::Arith(Arith::BitOr)),
            ("^", 6, Binary::Arith(Arith::BitXor)),
            ("&", 7, Binary::Arith(Arith::BitAnd)),
            ("==", 8, Binary::Compare(CompareKind::Eq)),
            ("!=", 8, Binary::Compare(CompareKind::Ne)),
            ("<", 9, Binary::Compare(CompareKind::Lt)),
            ("<=", 9, Binary::Compare(CompareKind::Le)),
            (">", 9, Binary::Compare(CompareKind::Gt)),
            (">=", 9, Binary::Compare(CompareKind::Ge)),
            ("<<", 10, Binary::Arith(Arith::Shl)),
            (">>", 10, Binary::Arith(Arith::Shr)),
            ("+", 11, Binary::Arith(Arith::Add)),
            ("-", 11, Binary::Arith(Arith::Sub)),
            ("*", 12, Binary::Arith(Arith::Mul)),
            ("/", 12, Binary::Arith(Arith::Div)),
            ("%", 12, Binary::Arith(Arith::Rem)),
        ];
        RULES
            .iter()
            .find(|(name, _, _)| *name == self.current().text)
            .map(|(_, bp, op)| (*bp, *bp + 1, *op))
    }

    fn expect_identifier(&mut self) -> diagnostic::Result<&'s str> {
        let token = self.current();
        if token
            .text
            .bytes()
            .next()
            .is_some_and(|byte| byte.is_ascii_alphabetic() || byte == b'_')
        {
            self.at += 1;
            return Ok(token.text);
        }
        Err(error(
            token.line,
            format!("期望标识符，得到 `{}`", token.text),
        ))
    }
}

fn parameter_type(element: ScalarType, pointers: u8, constant: bool) -> ValueType {
    if pointers == 0 {
        return ValueType::Scalar(element);
    }
    ValueType::Buf {
        element,
        access: if constant {
            Access::Read
        } else {
            Access::ReadWrite
        },
    }
}

fn register_function(
    function: Function,
    expected: &str,
    helpers: &mut Vec<Function>,
    main: &mut Option<Function>,
) -> diagnostic::Result<()> {
    if function.name != expected {
        helpers.push(function);
        return Ok(());
    }
    if function.result != ScalarType::Void || main.is_some() {
        return Err(error(1, "主内核必须唯一且返回 void"));
    }
    *main = Some(function);
    Ok(())
}

fn binary_node(op: Binary, left: Expr, right: Expr) -> Expr {
    let (left, right) = (Box::new(left), Box::new(right));
    match op {
        Binary::Arith(op) => Expr::Binary { op, left, right },
        Binary::Compare(op) => Expr::Compare { op, left, right },
        Binary::Logical(op) => Expr::Logical { op, left, right },
    }
}

enum Step {
    Next(Expr),
    Done(Expr),
}

#[cfg(test)]
#[path = "../tests/parser_tests.rs"]
mod tests;
