##  barycentric coordinates

重心坐标系



顶点 

$$
p_a, p_b, p_c
$$, 内部一点可以表示为 

$$
p = ap_a+bp_b+cp_c \\ a+b+c=1
$$ 


$$
a=\frac{area(pp_bp_c)}{area(p_ap_bp_c)}, b=\frac{area(pp_ap_c)}{area(p_ap_bp_c)}, c=\frac{area(pp_ap_b)}{area(p_ap_bp_c)}
$$

<hr>

$$
\begin{aligned}
 \lambda_1=\frac{(y_2-y_3)(x-x_3)+(x_3-x_2)(y-y_3)}{\det(T)}&=\frac{(y_2-y_3)(x-x_3)+(x_3-x_2)(y-y_3)}{(y_2-y_3)(x_1-x_3)+(x_3-x_2)(y_1-y_3)}\, \\
\lambda_2=\frac{(y_3-y_1)(x-x_3)+(x_1-x_3)(y-y_3)}{\det(T)}&=\frac{(y_3-y_1)(x-x_3)+(x_1-x_3)(y-y_3)}{(y_2-y_3)(x_1-x_3)+(x_3-x_2)(y_1-y_3)}\, \\
\lambda_3=1-\lambda_1-\lambda_2&=\frac{(y_1-y_2)(x-x_1) + (x_2-x_1)(y-y_1)}{(y_2-y_3)(x_1-x_3)+(x_3-x_2)(y_1-y_3)} 
\end{aligned}
$$

能构成三角形则分子不为零；

<hr>

### x, y 方向 $\Delta z$
 
$$
dx = \frac{(y_1-y_2)(z_2-z_3)-(y_2-y_3)(z_1-z_2)}{(x_2-x_3)(y_1-y_2)-(x_1-x_2)(y_2-y_3)} = \frac{(y_2-y_1)(z_2-z_3)+(y_2-y_3)(z_1-z_2)}{(y_2-y_3)(x_1-x_3)+(x_3-x_2)(y_1-y_3)} \\
dy = \frac{(x_1-x_2)(z_2-z_3) - (x_2-x_3)(z_1-z_2)}{(x_1-x_2)(y_2-y_3)- (x_2-x_3)(y_1-y_2)} = \frac{(x_1-x_2)(z_2-z_3) + (x_3-x_2)(z_1-z_2)}{(y_2-y_3)(x_1-x_3)+(x_3-x_2)(y_1-y_3)}
$$

能构成三角形则分子不为零；
透视投影后 $z$ 值仍然是线性的