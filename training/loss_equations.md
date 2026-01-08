### Loss Functions

**(1)**  
$$
\mathcal{L}_{\text{tot}}
= (1-\alpha)\left(\lambda_1 \mathcal{L}_2 + \lambda_2 \mathcal{L}_{\text{fea}} + \lambda_3 \mathcal{L}_{\text{adv}}\right)
+ \alpha \mathcal{L}_{\text{ce}}
$$

**(2)**  
$$
\mathcal{L}_2
= \frac{1}{HWC}
\sum_{h=1}^{H}
\sum_{w=1}^{W}
\sum_{c=1}^{C}
\left(I_{gt}(h,w,c) - I_{sr}(h,w,c)\right)^2
$$

**(3)**  
$$
\mathcal{L}_{\text{ce}}(x,y)
= -x_y + \log \sum_i e^{x_i}
$$

**(4)**  
$$
\mathcal{L}_{\text{fea}} = \mathcal{L}_1 + \mathcal{L}_{\text{cos}}
$$

**(5)**  
$$
\mathcal{L}_1
= \left\lVert \hat{F}_{\text{real}} - \hat{F}_{\text{fake}} \right\rVert_1
$$

**(6)**  
$$
\mathcal{L}_{\text{cos}}
= 1 - \cos\left(\hat{F}_{\text{real}}, \hat{F}_{\text{fake}}\right)
$$

**(7)**  
$$
\mathcal{L}_D
= \mathcal{L}_{\text{BCE}}\left(D(z_{\text{real}}), 1\right)
+ \mathcal{L}_{\text{BCE}}\left(D(z_{\text{fake}}), 0\right)
$$

**(8)**  
$$
\mathcal{L}_{\text{BCE}}(x,y)
= -\left[y \log(\sigma(u)) + (1-y)\log(1-\sigma(u))\right]
$$

**(9)**  
$$
\sigma(u) = \frac{1}{1 + e^{-u}},
\quad
u \in \{ D(z_{\text{real}}), D(z_{\text{fake}}) \}
$$

**(10)**  
$$
z_{\text{real}} = \operatorname{concat}(I_{gt}, S_{gt}),
\quad
z_{\text{fake}} = \operatorname{concat}(I_{sr}, S_{\text{pred}})
$$

**(11)**  
$$
\mathcal{L}_{\text{adv}}
= \mathcal{L}_{\text{BCE}}\left(D(z_{\text{fake}}), 1\right)
$$
