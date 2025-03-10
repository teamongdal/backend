import json
import numpy as np
import matplotlib.pyplot as plt

def continuous_bump_function(x, k1=1, k2=1, A=-0.00059259):
    """
    Calculate the piecewise function f(x) for a given x, satisfying:
    
      - For x < 50: 
          f(x) = exp((x - 50)/k1)
          → f(50)=1, and for x<50 the function decays rapidly toward 0.
      
      - For 50 ≤ x ≤ 95:
          f(x) = 1 + A*(x - 50)*(x - 95)
          → f(50)=1 and f(95)=1, with the maximum at x=72.5 being exactly 1.3.
      
      - For x > 95:
          f(x) = exp(-(x - 95)/k2)
          → f(95)=1, and for x>95 the function decays rapidly toward 0.
    
    Parameters:
      k1, k2 : Decay constants for x<50 and x>95 respectively (set to 1 for a steep decay).
      A      : Quadratic curvature parameter for 50 ≤ x ≤ 95.
               Set to -0.00059259 so that f(72.5)=1.3.
    """
    x = np.array(x, ndmin=1)
    return np.piecewise(
        x,
        [x < 50, (x >= 50) & (x <= 95), x > 95],
        [lambda x: np.exp((x - 50)/k1),
         lambda x: 1 + A*(x - 50)*(x - 95),
         lambda x: np.exp(-(x - 95)/k2)]
    )

def get_final_price_from_json(json_file, product_code):
    """
    Read the JSON file and find the record with the given product_code.
    Extract the 'final_price' value, remove commas and the trailing '원', and return as float.
    """
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    for record in data:
        if record.get("product_code") == product_code:
            price_str = record.get("final_price")
            # Remove commas and the trailing "원"
            price_str = price_str.replace(",", "").replace("원", "").strip()
            return float(price_str)
    raise ValueError(f"Product code {product_code} not found in {json_file}")

def get_final_price(value):
    # Remove commas and the trailing "원"
    price_str = value.replace(",", "").replace("원", "").strip()
    return float(price_str)


if __name__ == "__main__":
    # 테스트용 product_code (예시)
    product_code_dupe = "manual_maxionepiece_0004"    # dupe용 JSON에 있는 상품 코드
    product_code_luxury = "manual_maxionepiece_0001"     # luxury용 JSON에 있는 상품 코드

    # JSON 파일 경로
    dupe_json = "ver2_cat&color&clip&pattern.json"
    luxury_json = "myeongpum_test_cat&color&clip&pattern.json"
    
    # 각 파일에서 final_price 추출 (dupe_price와 luxury_price)
    dupe_price = get_final_price_from_json(dupe_json, product_code_dupe)
    luxury_price = get_final_price_from_json(luxury_json, product_code_luxury)
    
    # dc_ratio 계산: ((luxury_price - dupe_price) / luxury_price) * 100
    dc_ratio_value = (luxury_price - dupe_price) / luxury_price * 100

    # continuous_bump_function을 이용해 최종 가중치(y 값) 계산
    selected_weight = continuous_bump_function(dc_ratio_value)[0]
    
    print(f"For product codes {product_code_dupe} (dupe) and {product_code_luxury} (luxury):")
    print(f"  dupe_price  = {dupe_price}")
    print(f"  luxury_price = {luxury_price}")
    print(f"  dc_ratio     = {dc_ratio_value:.2f}%")
    print(f"  Final weight (y value) = {selected_weight:.3f}")

    # 그래프 그리기: x축 범위는 35부터 110
    x_values = np.linspace(35, 110, 1000)
    y_values = continuous_bump_function(x_values)
    
    # Key points: (50,1)와 (95,1)
    key_points_x = [50, 95]
    key_points_y = [1, 1]
    
    plt.figure(figsize=(16, 12))
    plt.plot(x_values, y_values, label="Continuous Bump Function", color="green")
    plt.plot(key_points_x, key_points_y, "ro", label="Key Points (50,1) and (95,1)")
    
    # 추출한 dc_ratio에 해당하는 점 표시
    plt.plot(dc_ratio_value, selected_weight, "bs", markersize=10, 
             label=f"Set Price: dc_ratio={dc_ratio_value:.2f}%")
    plt.annotate(f'y = {selected_weight:.3f}', (dc_ratio_value, selected_weight),
                 textcoords="offset points", xytext=(10, -10),
                 arrowprops=dict(arrowstyle="->", color="blue"))
    
    # Key point 위치에 수직 dashed 선 추가
    for x in key_points_x:
        plt.axvline(x=x, color="gray", linestyle="--", alpha=0.5)
    
    plt.title("Golden Dupe Price Function\n(Exponential decay for x<50 & x>95, Quadratic for 50≤x≤95)")
    plt.xlabel("dc_ratio (%)")
    plt.ylabel("Dupe_weight")
    plt.xlim(35, 110)
    plt.ylim(0, 1.6)
    plt.legend()
    plt.grid(True)
    plt.show()
