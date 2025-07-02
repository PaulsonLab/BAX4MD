import numpy as np
import torch
import matplotlib.pyplot as plt
from botorch.models.transforms.outcome import Standardize
from fixed_noise_gp import get_gp, train_gp


class TaskHandler:
    def __init__(self, task_kwargs):
        self.task_kwargs = task_kwargs

    def algorithm(self, sample_Y, max_var_indices):
        pass

    def get_valid_indices(self, bax, X, sample_Y, pred_Ymean, pred_Yvar, X1_range, X2_range, max_var_indices):
        raise NotImplementedError

    def post_step(self, bax, **kwargs):
        pass


class LevelSetHandler(TaskHandler):
    def get_valid_indices(self, bax, X, sample_Y, pred_Ymean, pred_Yvar, X1_range, X2_range, max_var_indices):
        return self.algorithm(sample_Y, max_var_indices)

    def algorithm(self, sample_Y, max_var_indices):
        lb = self.task_kwargs["lb"]
        ub = self.task_kwargs["ub"]
        mask = (sample_Y > lb) * (sample_Y < ub)
        mask[max_var_indices, :] = 0
        valid_indices = mask.squeeze().nonzero(as_tuple=True)[0]
        return valid_indices

    def post_step(self, bax, **kwargs):
        mask = self.plot_level_set(bax, kwargs['pred_Ymean'], kwargs['pred_Yvar'], kwargs['sample_Y'], kwargs['X1'], kwargs['X2'])
        if bax.true_mask is not None:
            F1 = self.compute_F1(mask, bax.true_mask)
            print(f"F1 Score: {F1:.4f}")
            bax.F1_list.append(F1)

    def plot_level_set(self, bax, pred_Ymean, pred_Yvar, sample_Y, X1, X2):
        lb = self.task_kwargs["lb"]
        ub = self.task_kwargs["ub"]
        Y_mean_grid = pred_Ymean.detach().numpy().reshape(X1.shape)
        Y_var_grid = pred_Yvar.detach().numpy().reshape(X1.shape)
        Y_ps_grid = sample_Y.detach().numpy().reshape(X1.shape)

        # fig, axes = plt.subplots(1, 3, figsize=(14, 6), dpi=600)
        fig, axes = plt.subplots(1, 3, figsize=(14, 6))

        cp1 = axes[0].contourf(X1, X2, Y_mean_grid, cmap="Blues", levels=20)
        cbar1 = fig.colorbar(cp1, ax=axes[0])
        cbar1.ax.tick_params(labelsize=16)
        mask = (Y_mean_grid > lb) * (Y_mean_grid < ub)
        axes[0].scatter(X1[mask], X2[mask], color='#f46d43', marker='o', alpha=0.3, s=80)
        axes[0].scatter(bax.train_X[:bax.init_D_size, 0].detach().numpy(), bax.train_X[:bax.init_D_size, 1].detach().numpy(), color='black', marker='x', s=80)
        axes[0].scatter(bax.train_X[bax.init_D_size:-1, 0].detach().numpy(), bax.train_X[bax.init_D_size:-1, 1].detach().numpy(), color='#1a9850', marker='x', s=80)

        if bax.true_mask is not None:
            axes[0].scatter(X1[bax.true_mask], X2[bax.true_mask], color='#d6604d', marker='1', alpha=0.8, s=80)

        axes[0].set_xlabel(r"$x_1$", fontsize=20)
        axes[0].set_ylabel(r"$x_2$", fontsize=20)
        axes[0].set_title("Posterior Mean", fontsize=20)
        axes[0].tick_params(axis='both', labelsize=16)

        cp2 = axes[1].contourf(X1, X2, Y_ps_grid, cmap="Blues", levels=20)
        cbar2 = fig.colorbar(cp2, ax=axes[1])
        cbar2.ax.tick_params(labelsize=16)
        # axes[1].scatter(bax.train_X[-1, 0].detach().numpy(), bax.train_X[-1:, 1].detach().numpy(), color='#e7298a', marker='x')
        axes[1].set_xlabel(r"$x_1$", fontsize=20)
        axes[1].set_ylabel(r"$x_2$", fontsize=20)
        axes[1].set_title("Posterior Sample", fontsize=20)
        axes[1].tick_params(axis='both', labelsize=16)

        cp3 = axes[2].contourf(X1, X2, Y_var_grid, cmap="Purples", levels=20)
        cbar3 = fig.colorbar(cp3, ax=axes[2])
        cbar3.ax.tick_params(labelsize=16)
        # axes[2].scatter(bax.train_X[-1, 0].detach().numpy(), bax.train_X[-1:, 1].detach().numpy(), color='#e7298a', marker='x')
        axes[2].set_xlabel(r"$x_1$", fontsize=20)
        axes[2].set_ylabel(r"$x_2$", fontsize=20)
        axes[2].set_title("Posterior Variance", fontsize=20)
        axes[2].tick_params(axis='both', labelsize=16)
        plt.tight_layout()
        plt.show()
        
        return torch.from_numpy(mask).to(torch.float64)

    def compute_F1(self, mask, true_mask):
        mask = mask.bool()
        true_mask = true_mask.bool()
        TP = (mask & true_mask).sum().item()
        TN = (~mask & ~true_mask).sum().item()
        FP = (mask & ~true_mask).sum().item()
        FN = (~mask & true_mask).sum().item()
        return 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0.0


class ManifoldCrawlingHandler(TaskHandler):
    def get_valid_indices(self, bax, X, sample_Y, pred_Ymean, pred_Yvar, X1_range, X2_range, max_var_indices):
        tau = self.task_kwargs["tau"]
        valid_indices, df_dx1, max_idx_per_x2, dg_dx1, max_idx_per_x2_g = self.algorithm(bax, X, sample_Y, pred_Ymean, X1_range, X2_range, tau, max_var_indices)
        self._cached = (df_dx1, max_idx_per_x2, dg_dx1, max_idx_per_x2_g)
        return valid_indices

    def post_step(self, bax, **kwargs):
        df_dx1, max_idx_per_x2, dg_dx1, max_idx_per_x2_g = self._cached
        self.x1_star = self.plot_manifold(bax, df_dx1, max_idx_per_x2, dg_dx1, max_idx_per_x2_g, kwargs['X1'], kwargs['X2'], kwargs['X1_range'], kwargs['X2_range'])

    def algorithm(self, bax, X, sample_Y, pred_Ymean, X1_range, X2_range, tau, max_var_indices):
        mask = torch.zeros(len(X1_range)*len(X2_range), dtype=torch.bool)
        _, X1, X2 = bax.get_mesh_grid(X1_range, X2_range)
        dx1 = X1_range[1] - X1_range[0]
        df_dx1 = (sample_Y.reshape(len(X2_range), len(X1_range))[:, 2:] - sample_Y.reshape((len(X2_range), len(X1_range)))[:, :-2]) / (2 * dx1)
        df_dx1 = torch.nn.functional.pad(df_dx1, (1, 1), mode='replicate')
        neg_df_dx1 = - df_dx1
        max_idx_per_x2 = torch.argmax(neg_df_dx1, dim=1)
        
        dg_dX = torch.autograd.grad(pred_Ymean, X, torch.ones_like(pred_Ymean), create_graph=True)
        dg_dx1 = dg_dX[0][:, 0].reshape((len(X2_range), len(X1_range)))
        neg_dg_dx1 = - dg_dx1
        max_idx_per_x2_g = torch.argmax(neg_dg_dx1, dim=1)
        
        flat_indices = max_idx_per_x2 + torch.arange(len(X2_range))*len(X1_range)
        
        mask[flat_indices] = True
        mask = mask.unsqueeze(1)
        mask[max_var_indices] = 0
        max_val_per_x2 = torch.max(neg_df_dx1, dim=1).values
        
        low_grad_x2_mask = max_val_per_x2 < tau
        low_grad_indices = max_idx_per_x2[low_grad_x2_mask] + torch.arange(len(X2_range))[low_grad_x2_mask] * len(X1_range)
        mask[low_grad_indices] = 0
        
        valid_indices = mask.squeeze().nonzero(as_tuple=True)[0]
        return valid_indices, df_dx1, max_idx_per_x2, dg_dx1, max_idx_per_x2_g

    def plot_manifold(self, bax, df_dx1, max_idx_per_x2, dg_dx1, max_idx_per_x2_g, X1, X2, X1_range, X2_range):
        df_dx1_np = df_dx1.detach().numpy()
        vlim_f = np.max(np.abs(df_dx1_np))
        
        dg_dx1_np = dg_dx1.detach().numpy()
        vlim_g = np.max(np.abs(dg_dx1_np))
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        cp1 = axes[0].contourf(X1, X2, df_dx1.detach().numpy(), cmap="bwr", levels=20, vmin=-vlim_f, vmax=vlim_f)
        axes[0].plot(X1_range[max_idx_per_x2], X2_range, color='green')
        axes[0].scatter(bax.train_X[:bax.init_D_size, 0].detach().numpy(), bax.train_X[:bax.init_D_size, 1].detach().numpy(), color='black', marker='x', s=80)
        axes[0].scatter(bax.train_X[bax.init_D_size:, 0].detach().numpy(), bax.train_X[bax.init_D_size:, 1].detach().numpy(), color='#e7298a', marker='x', s=80)
        cbar1 = fig.colorbar(cp1, ax=axes[0])
        cbar1.ax.tick_params(labelsize=16)
        axes[0].set_xlabel(r"$x_1$", fontsize=20)
        axes[0].set_ylabel(r"$x_2$", fontsize=20)
        axes[0].set_title(r"Thompson Sample $\frac{\partial Rg}{\partial x_1}$", fontsize=20)
        axes[0].tick_params(axis='both', labelsize=16)
        
        cp2 = axes[1].contourf(X1, X2, dg_dx1.detach().numpy(), cmap="bwr", levels=20, vmin=-vlim_g, vmax=vlim_g)
        axes[1].plot(X1_range[max_idx_per_x2_g], X2_range, color='green')
        axes[1].scatter(bax.train_X[:bax.init_D_size, 0].detach().numpy(), bax.train_X[:bax.init_D_size, 1].detach().numpy(), color='black', marker='x', s=80)
        axes[1].scatter(bax.train_X[bax.init_D_size:, 0].detach().numpy(), bax.train_X[bax.init_D_size:, 1].detach().numpy(), color='#e7298a', marker='x', s=80)
        cbar2 = fig.colorbar(cp2, ax=axes[1])
        cbar2.ax.tick_params(labelsize=16)
        axes[1].set_xlabel(r"$x_1$", fontsize=20)
        axes[1].set_ylabel(r"$x_2$", fontsize=20)
        axes[1].set_title(r"Predicted $\frac{\partial Rg}{\partial x_1}$", fontsize=20)
        axes[1].tick_params(axis='both', labelsize=16)
        
        plt.tight_layout()
        plt.show()
        
        return X1_range[max_idx_per_x2_g]


class BAX:
    def __init__(self, init_X, init_Y, init_noise, true_mask=None):
        self.init_X = init_X
        self.init_Y = init_Y
        self.init_noise = init_noise
        self.init_D_size = self.init_X.shape[0]
        self.train_X = self.init_X.clone()
        self.train_Y = self.init_Y.clone()
        self.train_noise = self.init_noise.clone()
        self.true_mask = true_mask

    def run(self, N, X1_range, X2_range, simulator, task, task_kwargs):
        X_flat, X1, X2 = self.get_mesh_grid(X1_range, X2_range)
        max_var_indices = []
        self.F1_list = []
        task_handlers = {
            "level set": LevelSetHandler,
            "manifold crawling": ManifoldCrawlingHandler,
        }
        
        if task not in task_handlers:
            raise ValueError(f"Unknown task: {task}")
        
        self.task_handler = task_handlers[task](task_kwargs)
        
        for n in range(N):
            print(f"Iteration {n}/{N-1}")
            model = train_gp(get_gp(self.train_X, self.train_Y, self.train_noise))
            X = X_flat.clone().detach().requires_grad_(True)
            posterior = model.posterior(X)
            sample_Y = posterior.rsample(torch.Size([1])).squeeze(0)
            pred_Ymean = posterior.mean.squeeze()
            pred_Yvar = posterior.variance.squeeze()
            valid_indices = self.task_handler.get_valid_indices(self, X, sample_Y, pred_Ymean, pred_Yvar, X1_range, X2_range, max_var_indices)
            
            if valid_indices.numel() > 0:
                max_var_index = valid_indices[torch.argmax(pred_Yvar[valid_indices])]
            else:
                max_var_index = torch.argmax(pred_Yvar)
                print("Empty set! Using max variance.")
                
            next_X = X[max_var_index].unsqueeze(0)

            # Uncomment the following line to run random baseline
            # next_X = X[torch.randint(low=0, high=1020, size=(1,)).item()].unsqueeze(0)

            next_Y, next_noise = simulator.run(next_X)
            next_noise = next_noise.pow(2)
            max_var_indices.append(max_var_index.item())
            self.train_X = torch.cat([self.train_X, next_X], dim=0)
            self.train_Y = torch.cat([self.train_Y, next_Y], dim=0)
            self.train_noise = torch.cat([self.train_noise, next_noise], dim=0)
            
            self.task_handler.post_step(
                self, sample_Y=sample_Y, pred_Ymean=pred_Ymean, pred_Yvar=pred_Yvar, X1=X1, X2=X2,
                X1_range=X1_range, X2_range=X2_range
            )
            
        if task == "level set":
            return (self.F1_list, self.train_X) if self.F1_list else self.train_X
        
        elif task == "manifold crawling":
            return self.task_handler.x1_star, self.train_X

    def get_mesh_grid(self, X1_range, X2_range):
        X1, X2 = np.meshgrid(X1_range, X2_range)
        X1_flat = X1.flatten()
        X2_flat = X2.flatten()
        grid_points = np.column_stack((X1_flat, X2_flat))
        X = torch.from_numpy(grid_points).to(torch.float64)
        return X, X1, X2
