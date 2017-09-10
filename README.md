# CNN-Conv-BatchNorm-fusion
Caffe python script to fuse BatchNorm (and Scale) layers into previous Conv layers

Conv:
```
convout(N_i,C_out)=bias(C_out )+∑_(k=0)^(C_in-1)〖weight(C_out,k)*input(N_i,k)〗
```
BN:
```
bnout(N_i,C_out )=((convout(N_i,C_out)-μ))⁄√(〖ϵ+ σ〗^2 )
```
SC:
```
scout(N_i,C_out )=γ*bn_out(N_i,C_out)+β
```
Overall:
```
out(N_i,C_out )=γ*((bias(C_out )+∑_(k=0)^(C_in-1)〖weight(C_out,k)*input(N_i,k)〗-μ))⁄√(〖ϵ+ σ〗^2 )+β
```
To fold BN and SC into Conv we can convert the formula to get new bias and weights:
```
newbias=γ*(bias-μ)/√(〖ϵ+ σ〗^2 )+β
newweights=γ*weights/√(〖ϵ+ σ〗^2 )
```
