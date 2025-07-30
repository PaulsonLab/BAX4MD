import torch
import pandas as pd


class Simulator():
    def __init__(self, polymer):
        if polymer == 'diblock':
            self.data_table = pd.read_csv('../data/diblock/diblock.csv')
        elif polymer == 'triblock':
            self.data_table = pd.read_csv('../data/triblock/triblock.csv')

    
    def run(self, X):
        x1 = X[:, 0].item()
        x2 = X[:, 1].item()
        match = self.data_table[(self.data_table['x1'] == x1) & (self.data_table['x2'] == x2)]
        
        if match.empty:
            raise ValueError(f"Invalid input: (x1={x1}, x2={x2})")
            
        rg = match['rg_mean'].iloc[0]
        noise = match['rg_std'].iloc[0]

        return torch.tensor(rg).to(torch.float64).reshape(-1, 1), torch.tensor(noise).to(torch.float64).reshape(-1, 1)

    
    def get_init_data(self, n, random_state):
        sample = self.data_table.sample(n, random_state=random_state)
        init_X = torch.tensor(sample[['x1', 'x2']].values, dtype=torch.float64).reshape(n, 2)
        init_rg = torch.tensor(sample['rg_mean'].values, dtype=torch.float64).reshape(n, 1)
        init_noise = torch.tensor(sample['rg_std'].values, dtype=torch.float64).reshape(n, 1)
        return init_X, init_rg, init_noise