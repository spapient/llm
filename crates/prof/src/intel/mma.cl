/* cutile-sign: mma/scalar_tile */
/* mma: D[m,n] = A[m,k] × B[k,n]，标量三重循环；槽位 $0..$8 = D,A,B,m,n,k,lda,ldb,ldd */
{
    for (int _i = 0; _i < ($3); ++_i) {
        for (int _j = 0; _j < ($4); ++_j) {
            float _acc = 0.0f;
            for (int _t = 0; _t < ($5); ++_t) {
                _acc += ($1)[_i * ($6) + _t] * ($2)[_t * ($7) + _j];
            }
            ($0)[_i * ($8) + _j] = _acc;
        }
    }
}
