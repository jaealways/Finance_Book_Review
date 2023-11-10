# [AP] Chapter 1. Consumption-Based Model and Overview

- 투자자는 소비, 저축, 투자 등의 결정을 내려야 하는데, pricing equation을 통해 이를 할 수 있음
- 오늘 포기하는 소비의 한계효용손실과 이로 인해 얻을 수 있는 내일의 한계효용이익과 일치해야 함
- 결국, 투자자의 한계효용을 통해 자산의 payoff를 할인해서 자산 가격을 계산하고, 이를 통해 많은 금융의 이슈들을 해결하고자 함
<br><br>
- 이자율은 한계효용 증가율의 기대값과 소비 경로 기대값에 영향을 미침
- 또한, 자산 가격에 대한 risk correction은 자산 payoff와 한계효용의 공분산, 자산 payoff와 소비의 공분산을 통해 이뤄져야 함 (분산이 아니라 공분산이 중요)
- 한계효용은 내가 어떻게 느끼는지 측정하는 방법인데, 자산 가격 결정 이론 대부분은 한계효용에 기반해서 논리가 전개.


# 1.1 Basic Pricing Equation

$$p_t = E_t [ \beta \frac{u'(c_{t+1})}{u'(c_t)}x_{t+1}]$$


- 투자자의 1차 조건을 통해 위의 식처럼 소비 기반 모델을 도출할 수 있음
- 결국, 불확실한 현금흐름을 가치평가하는 것이 중요한데, 아래와 같은 예를 들어보고자 함

- t 시점에 보상 $x_{t+1}$를 찾고 싶은데, 오늘의 주식 payoff(주가 + 배당)는 $x_{t+1} = p_{t+1} + d_{t+1}$로 나타낼 수 있음
- 이 때 $x_{t+1}$은 무작위변수임. 과연 이것이 일반 투자자에게 무슨 의미가 있을까? 
- 소비의 현재가치와 미래가치를 효용함수를 통해 정의하면 아래와 같음.
<!-- 갑자기? -->

$$U(c_t ,c_{t+1}) = u(c_t) + βE_t [u(c_{t+1})]$$

- $c_t$는 t 시점의 소비를 나타내는데, 이를 power function의 형태로 정의할 수 있음. (식 생략)
<!-- power function의 형태로 나타내는게 왜 중요?? concave한 모양 때문에?? -->
- 효용함수는 포트폴리오 수익, 분산 보다 소비에 대한 근본적인 욕구를 더 반영헀음
- $x_{t+1}$처럼 $c_{t+1}$도 랜덤이기 때문에, 투자자는 내일 그의 부와 소비량을 정확히 할 수 없음
- 위의 식을 통해 투자자의 위험회피와 조급함을 반영할 수 있음. 이 때 $\beta$는 인내심을 나타내는 주관적 할인요소임.
<br><br>

- 만약 투자자가 $x_{t+1}$의 payoff 자산을 $p_t$에 자유롭게 매수매도할 수 있다면, 얼마가 적절한가?
- $e$를 투자자가 원래 소비하는 수준, $ξ$를 자산을 구입량이라 하면, 아래 식으로 가격을 나타낼 수 있음

$$\textbf{max}\underset{\{ξ\}} u(c_t) + E_t[βu(c_{t+1})] s.t. \\ 
c_t = e_t − p_t ξ , \\
c_{t+1} = e_{t+1} + x_{t+1}ξ . $$

- 이 때, ξ를 0에 가까이 보내면, 소비와 포트폴리오 균형 사이에서 아래와 같은 균형가격을 구할 수 있음

$$
p_t=E_t\left[\beta \frac{u^{\prime}\left(c_{l+1}\right)}{u^{\prime}\left(c_t\right)} x_{t+1}\right] .
$$

- 결국 투자자는 한계 손실이 한계 이득과 동일할 때까지 자산을 계속 구입하거나 판매함
- 위 식을 통해 payoff $x_{t+1}$과 투자자의 소비 선택 ct, ct+1이 주어졌을 때, pt를 계산할 수 있음
- 위의 내용은 최적 소비와 포트폴리오를 고르기 위한 1차 조건으로, 자산 가격 결정 이론의 대부분은 이를 따름
<br><br>

- 위의 모형을 좀 더 근본적으로 파고 들면, 변수 e도 식에 포함시킬 수 있는데, CH 1.2처럼 유용한 예측을 얻을 수도 있음


 # 1.2 Marginal Rate of Substitution/Stochastic Discount Factor

$$
\begin{gathered}
p=E(m x), \\
m=\beta \frac{u^{\prime}\left(c_{t+1}\right)}{u^{\prime}\left(c_t\right)}
\end{gathered}
$$

- 소비 기반의 가격 결정 방정식을 위처럼 분해할 수 있는데, 이 때 $m_{t+1}$을 stocastic discount factor라 할 수 있음

$$
m_{t+1} \equiv \beta \frac{u^{\prime}\left(c_{t+1}\right)}{u^{\prime}\left(c_l\right)}
$$

- 이를 식 1.2에 다시 적용하면

$$
p_t=E_l\left(m_{t+1} x_{t+1}\right) .
$$

- t 시점의 가격, t+1 시점의 보상과 기대값은 t 시점의 정보의 조건부 값이 됨
- stochastic discount factor는 m이 불확실한 상황에서 어떻게 일반화할 수 있는지 나타냄
- 불확실성이 없다면 아래처럼 표현할 수 있음

$$
p_t=\frac{1}{R^f} x_{t+1} \text {, }
$$

- $R^f$가 무위험이자율이고, $1/R^f$는 할인요소가 되는데, 위험할수록 무위험자산보다 낮은 가격에 판매됨

$$
p_t^i=\frac{1}{R^i} E_t\left(x_{t+1}^i\right)
$$

- 이 때, 자산 고유의 위험 조정 할인율을 적용한다는 의미에서 i로 대체함
- 엄밀히 말하면, $m_{t+1}$은 t 시점에 확신할 수 없기 때문에, 확률적이거나 무작위성을 가짐
- 공통 할인요소 m과 자산 고유의 소득인 xi의 상관관계를 통해 자산 특유의 risk correction을 만들 수 있음
- $m_{t+1}$을 한계 대체율이라고도 하는데, 즉 투자자가 t 시점에 소비를 t+1 시점으로 얼마나 대체하고자 하는지 나타냄.


 # 1.3 Prices, Payoffs, and Notation

$$
\begin{array}{rcc} 
& \text { Price } p_t & \text { Payoff } x_{t+1} \\
\hline \text { Stock } & p_t & p_{t+1}+d_{t+1} \\
\text { Return } & 1 & R_{t+1} \\
\text { Price-dividend ratio } & \frac{p_t}{d_t} & \left(\frac{p_{t+1}}{d_{t+1}}+1\right) \frac{d_{t+1}}{d_t} \\
\text { Excess return } & 0 & R_{t+1}^e=R_{t+1}^a-R_{t+1}^b \\
\text { Managed portfolio } & z_t & z_t R_{t+1} \\
\text { Moment condition } & E\left(p_t z_t\right) & x_{t+1} z_t \\
\text { One-period bond } & p_t & 1 \\
\text { Risk-free rate } & 1 & R^f \\
\text { Option } & C & \max \left(S_T-K, 0\right)
\end{array}
$$

- 자산별 price와 payoff를 나타내면 위와 같음. 특히 주식, 채권, 옵션 등 다양한 자산에 하나의 이론을 적용할 수 있음을 확인할 수 있음
- 주식 gross return은 아래의 식으로 나타낼 수 있음

$$
R_{t+1} \equiv \frac{x_{t+1}}{p_t}
$$

- return을 가격이 1일 때의 소득으로 쉽게 생각해보면, 아래의 식을 따름

$$
1=E(m R)
$$

- 이 때 net return $r$은 $r=R-1$이나 $r=ln(R)$로 나타낼 수 있음 

<!-- stationary?? -->

$$
x_{t+1}=\left(1+\frac{p_{t+1}}{d_{t+1}}\right) \frac{d_{t+1}}{d_t}
$$

- 모든 상황에서 수익률이라는 컨셉이 필요한 건 아닌데, 가령 무위험이자율 $R^f$로 돈을 빌리고, $R$ 수익률의 자산에 투자한다면, 0의 자산으로 $R-R^f$의 현금흐름을 이끌어냄. (zero price)
- 이런 식으로 원금 투자없이 excess return $R^e$를 만들어내는 포트폴리오를 zero-cost 포트폴리오라 함

- 어떤 신호에 따라 투자하는 포트폴리오가 있다고 하면, t 시점에 투자되는 금액인 $z_t$는 $z_t R_{t+1}$만큼의 수익을 거둠
<!-- 가격과 수익률에 대한 비교가 계속 나오는 것 같은데 정확히 취지를 이해하지 못함 -->

- 가격과 수익률은 실질과 명목 둘 다 일수 있는데, 명목이라면 명목 할인요소를 사용해야 함. 가령 실제 가격과 소득을 아래와 같이 나타낼 수 있음

$$
\frac{p_t}{\Pi_t}=E_t\left[\left(\beta \frac{u^{\prime}\left(c_{t+1}\right)}{u^{\prime}\left(c_t\right)}\right) \frac{x_{t+1}}{\Pi_{t+1}}\right],
$$

- $\Pi$는 가격 수준을 나타냄. 위의 상황들을 종합했을 때, price를 $p_t$, payoff를 $x_{t+1}$로 나타내고자 함


# 1.4 Classic Issues in Finance

- 가격 방정식을 간단히 조작해서, 이자율, 리스크 조정, 평균분산 프론티어 등의 금융의 고전적인 이슈들을 소개하고자 함

## Risk-Free Rate

- 무위험이자율은 다음과 같이 정의됨 
$$
R^f=1 / E(m)
$$

- lognormal 소비성장과 파워 모양의 효용함수를 통해 아래 식을 유도할 수 있음
$$
r_t^f=\delta+\gamma E_t\left(\Delta \ln c_{t+1}\right)-\frac{\gamma^2}{2} \sigma_t^2\left(\Delta \ln c_{t+1}\right)
$$
- 실질이자율은 사람들이 인내심이 없고($δ$), 기대소비성장이 높거나, 리스크가 낮을 때 높음
- 효용함수가 더 곡선이면($\gamma$), 이자율이 기대소비성장에 더 민감하게 반응함 
<br><br>

- 무위험증권이 거래되지 않으면, 
<!-- ?? 무슨 취지인지 잘 모르겠음. 불확실성을 없애려고 하는 것? -->


- 세 가지 효과를 바로 확인할 수 있음
    - 사람들이 참을성이 없을 때, 즉 β가 낮을 때 실질 이자율은 높음
    - 실질이자율은 소비증가율이 높음
    - 실질금리는 파워 파라미터 γ가 크면 소비성장에 더 민감하게 반응

불확실성이 있을 때 이자율이 어떻게 작용하는지 이해하기 위해, 소비 증가율을 로그 정규 분포로 함, 이 때 실질 무위험 이자율 방정식은 아래와 같음

$$
r_t^f=\delta+\gamma E_t\left(\Delta \ln c_{t+1}\right)-\frac{\gamma^2}{2} \sigma_t^2\left(\Delta \ln c_{t+1}\right)
$$

- 아래와 같이 notation을 정의함

$$
r_t^f=\ln R_t^f ; \quad \beta=e^{-\delta}, \\
\Delta \ln c_{t+1}=\ln c_{t+1}-\ln c_t
$$

- 결국 취합 후에, 아래와 같은 식을 얻음

$$
R_t^f=\left[e^{-\delta} e^{-\gamma E_t\left(\Delta \ln c_{t+1}\right)+\left(\gamma^2 / 2\right) \sigma_t^2\left(\Delta \ln c_{t+1}\right)}\right]^{-1}
$$

- 로그노말 분포와 파워 효용함수를 혼합해서 1.7과 같은 결과를 얻음
- $σ^2$은 사전 예방적 저축을 의미하는데, 소비의 변동성이 클 때 소비가 낮을 때의 걱정이 클 때의 기쁨을 압도하기 때문에, 이자율이 더 높아짐

<!-- ?? risk aversion과 관련? -->


## Risk Corrections

- 소비증가와 소득이 양의 상관관계를 가질수록, 리스크를 보상받기 위해 가격은 낮아짐

$$
\begin{aligned}
p & =\frac{E(x)}{R^f}+\operatorname{cov}(m, x), \\
E\left(R^i\right)-R^f & =-R^f \operatorname{cov}\left(m, R^i\right) .
\end{aligned}
$$

- 공분산의 정의를 이용해서, 아래와 같은 식을 사용할 수 있음

$$
p=E(m) E(x)+\operatorname{cov}(m, x) .
$$

- 식 1.6을 이용해서 아래처럼 나타낼 수 있음
$$
p=\frac{E(x)}{R^f}+\operatorname{cov}(m, x)
$$

- 위 식의 첫 항은 현재가치에 관한 식이고, 두번 째 항은 리스크 조정을 나타내는 항임.
- 리스크 조정을 좀 더 자세히 이해하기 위해선, 아래의 식을 살펴볼 필요가 있음

$$
p=\frac{E(x)}{R^f}+\frac{\operatorname{cov}\left[\beta u^{\prime}\left(c_{t+1}\right), x_{t+1}\right]}{u^{\prime}\left(c_t\right)}
$$

- c가 증가함에 따라 한계효용이 증가함. 때문에, 자산의 소득이 소비와 양의 상관관계를 가지면 가격이 하락.
- 좀 더 구체적으로 말하면, 투자자들은 소비에 대한 불확실성을 좋아하지 않기 때문에, 소비를 조금 할 때 자산으로부터의 소득이 낮다면, 불확실성이 커지기 때문에 그 자산을 구입하고 싶은 니즈가 떨어짐.
- 이를 보험의 예로 들 수 있는데, 보험은 자산의 소득과 소비가 반대로 작동하기 때문에(극단적으로 집이 다 타버려서 소비가 음수일 때 자산의 소득이 생김), 보험의 가격이 보상 기대값보다 비싸더라도, 보험을 들고자 함
- 즉 분산보다는 소득과의 공분산이 리스크를 결정한다는 점에서, 투자자는 개별 자산이나 포트폴리오의 변동성보다 소비의 변동성에 더 큰 관심을 갖게 됨
- 투자자가 보상 x에 대해 ξ만큼 소량을 구매했을 때, 소비 변동성은 아래처럼 나타낼 수 있음

$$
\sigma^2(c+\xi x)=\sigma^2(c)+2 \xi \operatorname{cov}(c, x)+\xi^2 \sigma^2(x)
$$

<!--  그래서 결론?? 여기서 갑자기 asset pricing으로 넘어가는 논리? -->

- 자산가격결정모형에서 공분산 분해를 사용하면 아래와 같은 결과를 얻을 수 있음

$$
1=E(m) E\left(R^i\right)+\operatorname{cov}\left(m, R^i\right) \\
E\left(R^i\right)-R^f=-R^f \operatorname{cov}\left(m, R^i\right)
$$

- 모든 자산은 무위험이자율과 동일한 기대수익률을 갖고, 위험조정을 포함
- 수익률이 소비와 양의 상관관계를 갖는 자산은 기대수익률보다 더 높은 수익률을 보장해야 하고, 보험처럼 반대의 경우엔 더 낮은 수익률을 제공해도 됨


## Idiosyncratic Risk Does Not Affect Prices

- 할인요소와 완벽하게 상관성을 갖는 소득 성분만이 추가수익(프리미엄)을 냄

- 변동성이 큰 payoff를 갖는다고 큰 프리미엄이 부여되는게 아니라, payoff가 할인계수 m과 상관관계를 가져야 리스크 조정을 가질 수 있음

