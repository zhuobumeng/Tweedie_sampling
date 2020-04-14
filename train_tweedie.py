import pickle
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
import inspect
import argparse
import logging


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def auto_init_args(init):
    def new_init(self, *args, **kwargs):
        arg_dict = inspect.signature(init).parameters
        arg_names = list(arg_dict.keys())[1:]  # skip self
        proc_names = set()
        for name, arg in zip(arg_names, args):
            setattr(self, name, arg)
            proc_names.add(name)
        for name, arg in kwargs.items():
            setattr(self, name, arg)
            proc_names.add(name)
        remain_names = set(arg_names) - proc_names
        if len(remain_names):
            for name in remain_names:
                setattr(self, name, arg_dict[name].default)
        init(self, *args, **kwargs)

    return new_init


class horseshoe_fns():
    def __init__(self, tau, lam, *args, **kwargs):
        self.tau = tau
        self.lam = lam

    def pi_fn(self, u, tau=None, *args, **kwargs):
        if tau is None:
            tau = self.tau
        return u**(-1/2)*(1+u/(tau**2))**(-1)*(tau**(-1))

    def temp_fn(self, u, x, lam, tau, *args, **kwargs):
        return (lam+u)**(-3/2)*np.exp(-(x**2)/(2*(u+lam)))*self.pi_fn(u, tau)

    def fun_postmean(self, x, lam=None, tau=None, *args, **kwargs):
        if tau is None:
            tau = self.tau
        if lam is None:
            lam = self.lam
        if np.isscalar(x):
            x_list = [x]
        else:
            x_list = x
        res = []
        for x_ in x_list:
            A, _ = quad(lambda u:self.temp_fn(u, x_, lam, tau)*u, 0, np.infty)
            B, _ = quad(lambda u:self.temp_fn(u, x_, lam, tau)*(lam+u), 0, np.infty)
            res.append(A / B * x_)
        if np.isscalar(x):
            return res[0]
        else:
            return np.array(res)


class bayes_sparse_linear():
    @auto_init_args
    def __init__(self, n, num_iter, h, lam, mtd, s=10, A=None, x0=None, y=None, snr=10, *args, **kwargs):
        if A is None:
            self.A = self.generate_design(n)
        self.AA = np.dot(self.A.T, self.A)
        self.x = np.array([0] * s + [snr * np.sqrt(2*np.log(n)/n)] * (n - s))
        if y is None:
            self.y = self.generate_y(self.A, self.x)
        self.Ay = np.dot(self.A.T, self.y)
        if x0 is None:
            self.x0 = (np.dot(np.linalg.inv(self.AA), self.Ay)).reshape(-1)
        if mtd == "laplace":
            model = laplace_fns(rho=self.rho, lam=self.lam)
        else:
            model = horseshoe_fns(tau=self.tau, lam=self.lam)
        self.store, self.error_list, self.gradf_list = self.TDLMC(
            x0=self.x0, num_iter=num_iter, h=h, lam=lam,
            fun_postmean=model.fun_postmean, gradf=self.gradf)

    def generate_design(self, size):
        A = np.random.normal(size=(size, size))
        return A

    def generate_y(self, A, x):
        size = len(x)
        y = np.dot(A, x) + np.random.normal(size=size)
        return y

    def gradf(self, x, *args, **kwargs):
        return np.dot(self.AA, x) - self.Ay

    def TDLMC(self, x0, num_iter, h, lam, fun_postmean, gradf, view_num=2000, *args, **kwargs):
        store, error_list, gradf_list = [], [], []
        cur_x = np.array(x0)
        dim = len(cur_x)
        for i in range(num_iter):
            xi = np.random.normal(loc=0, scale=1, size=dim)
            post_mean = fun_postmean(cur_x)
            next_x = (1-h/lam)*cur_x - h*gradf(cur_x) + h/lam*post_mean + np.sqrt(2*h)*xi
            cur_x = next_x
            store.append(cur_x)
            error_list.append(((cur_x - self.x)**2).sum())
            gradf_list.append(gradf(cur_x))
            if i % view_num == 0:
                logger.info("iteration " + str(i) + " finished!")
                logger.info("error: " + str(error_list[-1]))
        return store, error_list, gradf_list


def run(config, logger):
    logger.info("*" * 25 + 'START' + "*" * 25)
    [A100, y100] = pickle.load(
        open("design_matrix/A_y_save.pkl", "rb"))

    config.tau = (config.s / config.n) / np.sqrt(config.n)
    config.h = config.lam
    np.random.seed(config.random_seed)
    x0 = np.random.normal(size=config.n)
    reg = bayes_sparse_linear(
        x0=x0, 
        A=A100, y=y100,
        num_iter=config.num_iter, h=config.h, mtd=config.mtd, lam=config.lam,
        n=config.n, s=config.s,
        tau=config.tau)

    pickle.dump([reg.h, reg.x0, reg.store, reg.error_list, reg.y, reg.gradf_list],
        open("res/seed" + str(config.random_seed) + "_niter" + str(config.num_iter) + \
            "h" + str(config.h) + "_x0random.pkl", "wb+"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.register('type', 'bool', str2bool)
    parser.add_argument('--n', type=int, default=50)
    parser.add_argument('--s', type=int, default=10)
    parser.add_argument("--lam", type=float, default=0.001)
    parser.add_argument('--num_iter', type=int, default=8000)
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument("--mtd", type=str, default="horseshoe")

    args = parser.parse_args()
    logging.basicConfig(
        filename="log/horseshoe" + "_n" + str(args.n) + "_seed" + \
            str(args.random_seed) + ".log",
        filemode='a+', level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%m-%d %H:%M')
    logger = logging.getLogger()

    run(config=args, logger=logger)
