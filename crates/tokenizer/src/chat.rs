use lib::tokenid;

use crate::error::tokenizererror;

/// A conversational role addressed by a chat template.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum chatrole {
    /// The system-instruction role.
    system,
    /// The user-turn role.
    user,
    /// The assistant-turn role.
    assistant,
}

impl chatrole {
    /// Returns the stable role name.
    #[must_use]
    pub const fn getname(self) -> &'static str {
        match self {
            Self::system => "system",
            Self::user => "user",
            Self::assistant => "assistant",
        }
    }
}

/// A single conversation message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct chatmessage {
    /// The role that produced the message.
    pub role: chatrole,
    /// The message text.
    pub content: String,
}

/// The prefix and suffix that wrap a given role's content.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct chattemplateentry {
    /// The role this entry formats.
    pub role: chatrole,
    /// Text emitted before the role's content.
    pub prefix: String,
    /// Text emitted after the role's content.
    pub suffix: String,
}

/// An explicit, non-Jinja chat template.
///
/// The template holds an optional beginning-of-sequence token and a per-role
/// prefix/suffix pair. Rendering concatenates each message's prefix, content,
/// and suffix; the result is then encoded by the tokenizer. This is a
/// deliberately small subset of the HuggingFace Jinja template language: it
/// expresses the common Llama-style role-wrapping format without evaluating
/// arbitrary template code.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct chattemplate {
    bos: Option<tokenid>,
    eos: Option<tokenid>,
    entries: Vec<chattemplateentry>,
}

impl chattemplate {
    /// Builds a template from an optional BOS/EOS token and role entries.
    #[must_use]
    pub fn new(
        bos: Option<tokenid>,
        eos: Option<tokenid>,
        entries: Vec<chattemplateentry>,
    ) -> Self {
        Self { bos, eos, entries }
    }

    /// Returns the beginning-of-sequence token, if configured.
    #[must_use]
    pub fn getbos(&self) -> Option<tokenid> {
        self.bos
    }

    /// Returns the end-of-sequence token, if configured.
    #[must_use]
    pub fn geteos(&self) -> Option<tokenid> {
        self.eos
    }

    /// Renders `messages` to a single prompt string.
    ///
    /// # Errors
    ///
    /// Returns [`tokenizererror::unsupportedrole`] when a message uses a role
    /// that has no entry in this template.
    pub fn render(&self, messages: &[chatmessage]) -> Result<String, tokenizererror> {
        let mut out = String::new();
        for message in messages {
            let entry = self
                .entries
                .iter()
                .find(|entry| entry.role == message.role)
                .ok_or(tokenizererror::unsupportedrole {
                    role: message.role.getname(),
                })?;
            out.push_str(&entry.prefix);
            out.push_str(&message.content);
            out.push_str(&entry.suffix);
        }
        Ok(out)
    }
}
