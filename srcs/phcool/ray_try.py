import ray

ray.init()

@ray.remote(num_cpus=14)
class calculate():
    def __init__(self,num) -> None:
        self.result=0
        self.num=num

    @ray.remote(num_cpus=1)
    def estimate_pi(num_terms_1,num_terms_2):
        pi_estimate = 0.0
        sign = 1
        for i in range(num_terms_1,num_terms_2):
            term = 1.0 / (2 * i + 1)
            pi_estimate += sign * term
            sign *= -1    
        return 4.0 * pi_estimate
    
    def get_result(self,data_nums):
        part=int(data_nums/(self.num))
        i=0.0
        a=[]
        for i in range(0,self.num):
            a.append(self.estimate_pi.remote(i*part,(i+1)*part))
        for i in range(0,self.num):
            mid=ray.get(a[i])
            self.result+=mid
        return self.result
        

Actor=calculate.remote(14)
result=Actor.get_result.remote(1000000000)
b=ray.get(result)
print(b)
print("\n\n")