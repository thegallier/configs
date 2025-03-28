Summary: Convex Polynomial Price Adjustment Optimization

Objective:
To develop and analyze a price adjustment model for trades based on customer tier and trade DV01. The model utilizes convex and monotonic Bernstein polynomials, allowing for flexible yet controlled adjustments. The goal is to optimize the scaling of these polynomial adjustments to maximize the Profit & Loss (P&L) generated from winning trades, subject to a user-defined constraint on the Losing DV01 Ratio.

Methodology:

Input Data:

Trade data including customerName, tier, firmAccount, cusip, amount, mid price, execution side (BUY/SELL), tradePrice, and dv01.

A per-cusip sensitivity factor epsilon (fixed at 0.1 in this implementation).

Polynomial Modeling:

Two 1-dimensional Bernstein polynomials are used as the basis for adjustments:

f_1(t): Based on normalized customer tier (tier_{norm} \in [0, 1]).

f_2(d): Based on normalized dv01 (dv01_{norm} \in [0, 1]).

The polynomials are defined by their control points (coefficients) 
C
k
(
1
)
C 
k
(1)
​
 
 and 
C
k
(
2
)
C 
k
(2)
​
 
, respectively.

Two rescaling factors, 
r
1
r 
1
​
 
 and 
r
2
r 
2
​
 
, are introduced to globally scale the output of the base polynomials.

Polynomial Fitting (Base Coefficients):

The user provides target points (4 initial Z-values for each polynomial) via sliders. These define target shapes.

The base polynomial coefficients (
C
k
(
1
)
C 
k
(1)
​
 
, 
C
k
(
2
)
C 
k
(2)
​
 
) are determined by solving separate optimization problems for each polynomial:

Objective: Minimize the sum of squared errors between the polynomial output and the user-defined target Z-values at the corresponding normalized X-coordinates.

Constraints:

Convexity: Enforced for both polynomials (
Δ
2
C
≥
0
Δ 
2
 C≥0
). This ensures the second derivative is non-negative.

Monotonicity (Increasing): Enforced only for the DV01 polynomial (
f
2
f 
2
​
 
) (
Δ
C
≥
0
ΔC≥0
). This ensures a larger DV01 results in a larger (or equal) multiplier basis.

Solver: CVXPY with a suitable solver (e.g., SCS).

Price Adjustment Simulation:

For each trade i:

Normalize 
t
i
e
r
i
tier 
i
​
 
 and 
d
v
0
1
i
dv01 
i
​
 
.

Calculate base polynomial outputs: 
v
1
,
b
a
s
e
=
f
1
(
t
i
e
r
n
o
r
m
,
i
)
v 
1,base
​
 =f 
1
​
 (tier 
norm,i
​
 )
, 
v
2
,
b
a
s
e
=
f
2
(
d
v
0
1
n
o
r
m
,
i
)
v 
2,base
​
 =f 
2
​
 (dv01 
norm,i
​
 )
.

Apply rescaling: 
v
a
l
u
e
1
=
r
1
⋅
v
1
,
b
a
s
e
value 
1
​
 =r 
1
​
 ⋅v 
1,base
​
 
, 
v
a
l
u
e
2
=
r
2
⋅
v
2
,
b
a
s
e
value 
2
​
 =r 
2
​
 ⋅v 
2,base
​
 
.

Calculate adjustment: 
a
d
j
u
s
t
m
e
n
t
i
=
ϵ
i
⋅
v
a
l
u
e
1
⋅
v
a
l
u
e
2
⋅
s
i
d
e
_
s
i
g
n
i
adjustment 
i
​
 =ϵ 
i
​
 ⋅value 
1
​
 ⋅value 
2
​
 ⋅side_sign 
i
​
 
 (where 
s
i
d
e
_
s
i
g
n
side_sign
 is -1 for BUY, +1 for SELL).

Calculate adjusted price: 
a
d
j
_
p
r
i
c
e
i
=
m
i
d
i
+
a
d
j
u
s
t
m
e
n
t
i
adj_price 
i
​
 =mid 
i
​
 +adjustment 
i
​
 
.

Metrics Calculation:

Win Condition: A trade i is 'winning' if 
a
d
j
_
p
r
i
c
e
i
>
t
r
a
d
e
P
r
i
c
e
i
adj_price 
i
​
 >tradePrice 
i
​
 
 for BUYs, or 
a
d
j
_
p
r
i
c
e
i
<
t
r
a
d
e
P
r
i
c
e
i
adj_price 
i
​
 <tradePrice 
i
​
 
 for SELLs. Otherwise, it's 'losing'.

Losing DV01 Ratio: 
(
∑
i
∈
Losing
d
v
0
1
i
)
/
(
∑
all 
i
d
v
0
1
i
)
(∑ 
i∈Losing
​
 dv01 
i
​
 )/(∑ 
all i
​
 dv01 
i
​
 )
.

Potential P&L (per trade): 
p
o
t
_
p
n
l
i
=
a
m
o
u
n
t
i
⋅
(
m
i
d
i
−
a
d
j
_
p
r
i
c
e
i
)
⋅
p
n
l
_
s
i
g
n
i
pot_pnl 
i
​
 =amount 
i
​
 ⋅(mid 
i
​
 −adj_price 
i
​
 )⋅pnl_sign 
i
​
 
 (where 
p
n
l
_
s
i
g
n
pnl_sign
 is +1 for BUY, -1 for SELL). This measures P&L versus mid.

Actual Winning P&L: 
∑
i
∈
Winning
p
o
t
_
p
n
l
i
∑ 
i∈Winning
​
 pot_pnl 
i
​
 
.

Potential P&L (All Trades): 
∑
all 
i
p
o
t
_
p
n
l
i
∑ 
all i
​
 pot_pnl 
i
​
 
.

P&L (Favorable Adjustment): 
∑
i
 where Adj Favors
p
o
t
_
p
n
l
i
∑ 
i where Adj Favors
​
 pot_pnl 
i
​
 
, where 'Adj Favors' means 
a
d
j
_
p
r
i
c
e
≤
m
i
d
adj_price≤mid
 for BUYs or 
a
d
j
_
p
r
i
c
e
≥
m
i
d
adj_price≥mid
 for SELLs.

Efficiency Ratio: 
A
c
t
u
a
l
W
i
n
n
i
n
g
P
N
L
/
P
o
t
e
n
t
i
a
l
P
N
L
(
A
l
l
T
r
a
d
e
s
)
ActualWinningPNL/PotentialPNL(AllTrades)
.

Optimization:

Goal: Find optimal rescaling factors 
r
1
∗
,
r
2
∗
r 
1
∗
​
 ,r 
2
∗
​
 
.

Objective: Maximize 
A
c
t
u
a
l
W
i
n
n
i
n
g
P
N
L
ActualWinningPNL
.

Constraint: 
∣
Losing DV01 Ratio
−
Target Ratio
∣
≤
Tolerance
∣Losing DV01 Ratio−Target Ratio∣≤Tolerance
. The 
T
a
r
g
e
t
R
a
t
i
o
TargetRatio
 is user-defined via an input field.

Bounds: 
0.5
≤
r
1
,
r
2
≤
2.0
0.5≤r 
1
​
 ,r 
2
​
 ≤2.0
.

Solver: scipy.optimize.minimize using the COBYLA method.

Visualization (Dashboard):

Interactive sliders for initial polynomial Z-values and rescaling factors (
r
1
r 
1
​
 
, 
r
2
r 
2
​
 
).

Input fields for polynomial degree and optimization Target Ratio.

Plots showing initial PWL points, current PWL points, the base fitted polynomial (convex/monotonic), and the rescaled polynomial. Axes are scaled to original Tier/DV01 ranges.

Display of global metrics (Losing DV01 Ratio, various P&Ls, Efficiency).

Heatmaps showing Losing DV01 Ratio and Efficiency Ratio across a grid of 
r
1
,
r
2
r 
1
​
 ,r 
2
​
 
 values.

Comparison tables showing initial vs. current Losing Rate and Winning PNL, aggregated by CUSIP, Tier, and Customer, with sortable delta columns.

Download button for the full results DataFrame including the adjusted price.

Key Formulas:

Bernstein Basis Polynomial:
B
k
,
n
(
t
)
=
(
n
k
)
t
k
(
1
−
t
)
n
−
k
,
t
∈
[
0
,
1
]
B 
k,n
​
 (t)=( 
k
n
​
 )t 
k
 (1−t) 
n−k
 ,t∈[0,1]

Polynomial Evaluation (Base):
f
(
t
n
o
r
m
)
=
∑
k
=
0
n
C
k
B
k
,
n
(
t
n
o
r
m
)
f(t 
norm
​
 )= 
k=0
∑
n
​
 C 
k
​
 B 
k,n
​
 (t 
norm
​
 )

Rescaled Values:
v
a
l
u
e
1
=
r
1
∑
k
=
0
n
1
C
k
(
1
)
B
k
,
n
1
(
t
i
e
r
n
o
r
m
)
value 
1
​
 =r 
1
​
  
k=0
∑
n 
1
​
 
​
 C 
k
(1)
​
 B 
k,n 
1
​
 
​
 (tier 
norm
​
 )

v
a
l
u
e
2
=
r
2
∑
k
=
0
n
2
C
k
(
2
)
B
k
,
n
2
(
d
v
0
1
n
o
r
m
)
value 
2
​
 =r 
2
​
  
k=0
∑
n 
2
​
 
​
 C 
k
(2)
​
 B 
k,n 
2
​
 
​
 (dv01 
norm
​
 )

Adjustment:
a
d
j
u
s
t
m
e
n
t
=
ϵ
⋅
v
a
l
u
e
1
⋅
v
a
l
u
e
2
⋅
sign
(
side
)
adjustment=ϵ⋅value 
1
​
 ⋅value 
2
​
 ⋅sign(side)

where sign
(
BUY
)
=
−
1
,
sign
(
SELL
)
=
+
1
where sign(BUY)=−1,sign(SELL)=+1

Adjusted Price:
a
d
j
_
p
r
i
c
e
=
m
i
d
+
a
d
j
u
s
t
m
e
n
t
adj_price=mid+adjustment

Convexity Constraint:
C
k
+
2
−
2
C
k
+
1
+
C
k
≥
0
∀
k
∈
{
0
,
.
.
.
,
n
−
2
}
C 
k+2
​
 −2C 
k+1
​
 +C 
k
​
 ≥0∀k∈{0,...,n−2}

Monotonicity Constraint (Increasing):
C
k
+
1
−
C
k
≥
0
∀
k
∈
{
0
,
.
.
.
,
n
−
1
}
C 
k+1
​
 −C 
k
​
 ≥0∀k∈{0,...,n−1}

Optimization Problem:
max
⁡
0.5
≤
r
1
,
r
2
≤
2.0
(
∑
i
∈
Winning
(
r
1
,
r
2
)
pot_pnl
i
(
r
1
,
r
2
)
)
0.5≤r 
1
​
 ,r 
2
​
 ≤2.0
max
​
  
​
  
i∈Winning(r 
1
​
 ,r 
2
​
 )
∑
​
 pot_pnl 
i
​
 (r 
1
​
 ,r 
2
​
 ) 
​
 

subject to:
∣
∑
i
∈
Losing
(
r
1
,
r
2
)
d
v
0
1
i
∑
all 
i
d
v
0
1
i
−
Target Ratio
∣
≤
Tolerance
​
  
∑ 
all i
​
 dv01 
i
​
 
∑ 
i∈Losing(r 
1
​
 ,r 
2
​
 )
​
 dv01 
i
​
 
​
 −Target Ratio 
​
 ≤Tolerance

Conclusion:
This approach provides a structured way to model price adjustments using constrained polynomials derived from simple user inputs. It allows for analysis of trade outcomes under different scaling assumptions and enables optimization to find parameters that best meet a defined objective (maximizing winning P&L) under a quantifiable risk constraint (Losing DV01 Ratio). The interactive dashboard facilitates exploration and understanding of the model's behavior.
