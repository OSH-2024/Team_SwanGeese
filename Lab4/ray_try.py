import ray
ray.init()

cpu_num=8
part_num=3



@ray.remote
def estimate_pia(num_terms_1,num_terms_2):
    pi_estimate = 0.0
    sign = 1
    for i in range(num_terms_1,num_terms_2):
        term = 1.0 / (2 * i + 1)
        pi_estimate += sign * term
        sign *= -1    
    return 4.0 * pi_estimate
    

def get_result(data_nums):
    part=int(data_nums/(part_num))
    result=0
    a=[]
    for i in range(0,part_num):
        a.append(estimate_pia.remote(i*part,(i+1)*part))
    for i in range(0,part_num):
        mid=ray.get(a[i])
        result+=mid
    return result
        

result=get_result(1000000000)
print(result)
print("\n\n")