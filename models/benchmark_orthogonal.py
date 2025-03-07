import torch 

from orthogonal import torch_householder_orgqr as householder_incumbent 
from torch_householder import torch_householder_orgqr as householder_challenger

NUM_TRIALS = 20 
d_min, d_max = 5, 15 
n_min, n_max = 128, 256 

if __name__ == '__main__': 
    data = list() 
    for _ in range(NUM_TRIALS): 
        d = torch.randint(d_min, d_max, size=(1,)).item() 
        n = torch.randint(n_min, n_max, size=(1,)).item()  
        params = torch.randn((n, d, d)) 
        eye = torch.eye(d).unsqueeze(0).repeat(params.size(0), 1, 1)
        A = params.tril(diagonal=-1) + eye
        
        tau = 2 / torch.linalg.norm(A, dim=2).pow(2).clamp(min=1e-12) 

        Q_c = householder_challenger(A, tau=tau)
        Q_i = householder_incumbent(A, tau=tau) 
        
        data.append(
            (
                torch.linalg.norm((Q_c - Q_i), dim=(1, 2)).mean() / torch.linalg.norm(Q_c, dim=(1, 2)).mean(), d, n  
            ) 
        )
    
    avg_err = sum(
        list(map(lambda x: x[0], data)) 
    ) / len(data) 
    print(avg_err) # tensor(2.0484e-07)

