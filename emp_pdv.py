import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from  scipy.optimize import least_squares, curve_fit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV, LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import root_mean_squared_error, r2_score
import yfinance as yf

dt = 1/252

def get_yf_data(symbol, start=None, end=None):
    """
    Download adjusted-close prices (or index levels) from Yahoo Finance.
 
    Parameters
    ----------
    symbol : str
        Yahoo Finance ticker, e.g. "^GSPC", "^VIX", "^NDX", "^VXN",
        "GLD", "^GVZ", "USO", "^OVX".
    start : str or None
        ISO date string, e.g. "1996-01-01".
    end : str or None
        ISO date string, e.g. "2022-05-15".
 
    Returns
    -------
    pd.DataFrame
        Columns: ["date", symbol]  — same shape as get_index_data().
    """
    ticker = yf.Ticker(symbol)
    hist = ticker.history(start=start, end=end, auto_adjust=True)
 
    if hist.empty:
        raise ValueError(f"yfinance returned no data for {symbol}")
 
    df = hist[["Close"]].copy()
    df = df.rename(columns={"Close": symbol})
    df.index = df.index.tz_localize(None)       # strip timezone
    df = df.reset_index().rename(columns={"Date": "date"})
    df["date"] = pd.to_datetime(df["date"])
 
    return df[["date", symbol]]

def get_index_data(symbol, suffix, start=None, end=None, api_key=None, filepath=None):
    if api_key is not None:
        url = f"https://eodhd.com/api/eod/{symbol}.{suffix}"
        params = {
            "fmt": "json",
            "api_token": api_key
        }
        if start is not None:
            params["from"] = start
        
        if end is not None:
            params["to"] = end

        r = requests.get(url, params=params)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
    
    else:
        df = pd.read_csv(filepath)

    if "adjusted_close" in df.columns:
        df.rename(columns={"adjusted_close": symbol}, inplace=True)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df[["date", symbol]]

def data_prep(index, vol, index_df, vol_df, max_delta=1000, forecast=0):

    df = pd.merge(index_df, vol_df, on="date")
    df.dropna(subset=[index], inplace=True)

    df["lag_0"] = df.loc[:, index] / df.loc[:, index].shift(1) - 1
    
    ret_values = df["lag_0"].values
    n = len(ret_values)
    
    lags = np.full((n, max_delta-1), np.nan)
    for lag in range(max_delta-1):
        lags[lag + 1:, lag] = ret_values[:n - lag - 1]
    
    lag_df = pd.DataFrame(lags, columns=[f"lag_{i+1}" for i in range(max_delta-1)], index=df.index)
    df = pd.concat([df, lag_df], axis=1)
    df.set_index("date", inplace=True)
    df[vol] = df[vol] / 100
    df[vol] = df[vol].shift(-forecast)

    return df.drop(columns=[index, vol]), df[vol]

def split_data(data, train_start, test_start, test_end):
    train_start = pd.to_datetime(train_start)
    test_start = pd.to_datetime(test_start)
    test_end = pd.to_datetime(test_end)
    X_train = data.loc[train_start:test_start]
    X_test = data.loc[test_start:test_end]
    return X_train.dropna(), X_test.dropna()

def get_data(
        index, vol, index_suffix, vol_suffix,
        load_from, train_start, test_start, test_end,
        max_delta, forecast=0,
        source="eodhd", api_key=None, filepath=None):
    
    if source == "eodhd":
        index_df = get_index_data(index, suffix=index_suffix, start=load_from, end=test_end, api_key=api_key)
        vol_df = get_index_data(vol, suffix=vol_suffix, start=load_from, end=test_end, api_key=api_key)
    elif source == "yf":
        index_df = get_yf_data(index, start=load_from, end=test_end)
        vol_df = get_yf_data(vol, start=load_from, end=test_end)
    elif source == "csv":
        index_df = get_index_data(index, suffix=index_suffix, start=load_from, end=test_end, filepath=filepath)
        vol_df = get_index_data(vol, suffix=vol_suffix, start=load_from, end=test_end, filepath=filepath)
    else:
        print("Invalid source please choose 'eodhd', 'yf' or 'csv'")
        index_df, vol_df = pd.DataFrame(), pd.DataFrame()
    
    
    index_wide, vol_wide = data_prep(index, vol, index_df, vol_df, max_delta=max_delta, forecast=forecast)

    X_train, X_test = split_data(index_wide, train_start, test_start, test_end)
    y_train, y_test = split_data(vol_wide, train_start, test_start, test_end)

    common_train_idx = X_train.index.intersection(y_train.index)
    X_train = X_train.loc[common_train_idx]
    y_train = y_train.loc[common_train_idx]

    common_test_idx = X_test.index.intersection(y_test.index)
    X_test = X_test.loc[common_test_idx]
    y_test = y_test.loc[common_test_idx]

    return X_train, X_test, y_train, y_test
    

def split_params(params, model_spec):
    n = len(model_spec)
    deltas = params[-n:]
    alphas = params[-2*n:-n]
    betas = params[1:-2*n]
    intercept = params[0]
    return intercept, betas, alphas, deltas

def comp_weighted_sum(x, weight_func, alpha, delta, power):
    dts = np.arange(x.shape[1]) * dt
    weights = weight_func(dts, alpha, delta)
    return np.sum(x**power*weights, axis=1)

def parse_feat(feat):
    if isinstance(feat, tuple):
        return feat[0], feat[1]
    return "raw", feat

def clip_returns(returns, sign):
    if sign == "pos":
        return np.maximum(returns, 0)
    if sign == "neg":
        return np.minimum(returns, 0)
    return returns

class TSPL:
    def __init__(self):
        self.num_params = 2

    def bounds(self, num_feat):
        num_params = 1 + num_feat * (self.num_params + 1)
        lower_bound = np.full(num_params, -np.inf)
        lower_bound[num_feat+1: 2*num_feat+1] = 0 # alphas >= 0
        lower_bound[2*num_feat+1:] = dt/100 # deltas >= 1/25200 > 0


        upper_bound = np.full(num_params, np.inf)
        upper_bound[num_feat+1: 2*num_feat+1] = 10 # alphas <= 10
        
        return lower_bound, upper_bound

    @staticmethod
    def kernel(tau, alpha, delta):
        return (tau + delta) ** (-alpha)
    
    @staticmethod
    def kernel_with_coef(tau, beta, alpha, delta):
        return beta * TSPL.kernel(tau, alpha, delta)
    
    @staticmethod
    def dk_dalpha(tau, alpha, delta):
        return -np.log(tau + delta) * TSPL.kernel(tau, alpha, delta)
    
    @staticmethod
    def dk_ddelta(tau, alpha, delta):
        return - alpha / (tau + delta) * TSPL.kernel(tau, alpha, delta)
    
    @staticmethod
    def norm_const(alpha, delta, max_delta, j):
        return np.sum(TSPL.kernel(np.arange(max_delta) * dt, alpha, delta) * dt) ** j
    
class NormTSPL:
    def __init__(self):
        self.num_params = 2

    def bounds(self, num_feat):
        num_params = 1 + num_feat * (self.num_params + 1)
        lower_bound = np.full(num_params, -np.inf)
        lower_bound[num_feat+1: 2*num_feat+1] = 0 # alphas >= 0
        lower_bound[2*num_feat+1:] = dt/100 # deltas >= 1/25200 > 0


        upper_bound = np.full(num_params, np.inf)
        upper_bound[num_feat+1: 2*num_feat+1] = 10 # alphas <= 10
        
        return lower_bound, upper_bound

    @staticmethod
    def kernel(tau, alpha, delta):
        tspl = (tau + delta) ** (-alpha) # TSPL kernel
        Z = delta ** (1-alpha) / (alpha-1) # (continuous) normalising constant
        return  tspl / Z
    
    @staticmethod
    def kernel_with_coef(tau, beta, alpha, delta):
        return beta * NormTSPL.kernel(tau, alpha, delta)
    
    @staticmethod
    def dk_dalpha(tau, alpha, delta):
        return (1/(alpha-1) - np.log(tau/delta+1)) * NormTSPL.kernel(tau, alpha, delta)

    @staticmethod
    def dk_ddelta(tau, alpha, delta):
        return ((alpha-1)/delta - alpha/(tau+delta)) * NormTSPL.kernel(tau, alpha, delta)
    
    @staticmethod
    def norm_const(*args, **kwargs):
        return 1 # already normalised

class PDVModel:
    def __init__(self, KernelClass, model_spec=(1, 2), max_delta=1000,
                 spans=[10, 20, 120, 250], cv_splits=0, use_lasso=True, plot=False,
                 neg_ret_feat=False, pos_ret_feat=False):
        self.Kernel = KernelClass()
        self.model_spec = model_spec
        self.num_feat = len(self.model_spec)
        self.max_delta = max_delta
        self.spans = np.array(spans)
        self.spans = self.spans[self.spans <= max_delta]
        if len(self.spans) < 2:
            self.spans = np.array([2, max_delta])
        self.cv_splits = cv_splits
        self.use_lasso = use_lasso
        self.plot = plot
        self.neg_ret_feat = neg_ret_feat
        self.pos_ret_feat = pos_ret_feat
        self.n_extra = neg_ret_feat + pos_ret_feat

    def comp_features(self, returns, alphas, deltas):
        dts = np.arange(returns.shape[1]) * dt
        features = {}
        
        for i, feat in enumerate(self.model_spec):
            sign, feat_power = parse_feat(feat)
            clipped_returns = clip_returns(returns, sign)

            kernel_weighted_sum = comp_weighted_sum(clipped_returns, self.Kernel.kernel, alphas[i], deltas[i], feat_power)
            features[feat] = np.sign(kernel_weighted_sum) * np.abs(kernel_weighted_sum) ** (1/feat_power)

        return features
    
    def lin_of_features(self, returns, params, ret_feats=False):

        lin_combo = 0
        if self.n_extra:
            extra_betas = params[-self.n_extra:]
            params = params[:-self.n_extra]
            idx = 0
            if self.neg_ret_feat:
                lin_combo += extra_betas[idx] * np.minimum(returns.iloc[:, 1].values, 0)
                idx += 1
            if self.pos_ret_feat:
                lin_combo += extra_betas[idx] * np.maximum(returns.iloc[:, 1].values, 0)

        intercept, betas, alphas, deltas = split_params(params, self.model_spec)
        features =self.comp_features(returns, alphas, deltas)
        lin_combo += intercept
        for i, feat in enumerate(self.model_spec):
            lin_combo += betas[i] * features[feat]            

        if ret_feats:
            return features, lin_combo

        return lin_combo

    def init_params(self, X, y):

        x = X.iloc[:, 0] # get returns series
        init_alphas, init_deltas = [], []

        cv = TimeSeriesSplit(n_splits=self.cv_splits) if self.cv_splits != 0 else None
        
        for feat in self.model_spec:

            sign, feat_power = parse_feat(feat)
            clipped_x = clip_returns(x, sign)

            ewma = {span: pd.Series.ewm(clipped_x ** feat_power, span=span).mean() for span in self.spans}
            X_ewma = pd.DataFrame.from_dict(ewma, orient="columns")

            if self.use_lasso:
                reg = LassoCV(
                    cv=cv,
                    max_iter=50000,
                    tol=1e-4,
                    positive=True
                )
                scaler = StandardScaler()
                X_fit = scaler.fit_transform(X_ewma)
            else:
                reg = RidgeCV(
                    cv=cv,
                    alphas = (0.1, 1.0, 10.0) if cv is None else np.logspace(-4, 4, 50),
                    scoring = None if cv is None else "r2"
                )
                scaler = None
                X_fit = X_ewma

            reg.fit(X_fit, y ** feat_power)
            coef_raw = reg.coef_
            coef = reg.coef_ / scaler.scale_ if scaler is not None else reg.coef_

            print(f"power={feat_power}: coefs={coef}, sum={coef.sum():.6f}")

            lambdas = 2 / (1 + self.spans)

            n_selected = np.count_nonzero(coef)
            if n_selected == 0:
                fallback_spans = np.array([s for s in [10, 20, 120, 250] if s <= self.max_delta])
                if len(fallback_spans) < 2:
                    fallback_spans = np.array([2, self.max_delta])
                print(f"Lasso zeroed everything, falling back to Ridge with spans={list(fallback_spans)}")
                ewma_fb = {s: pd.Series.ewm(clipped_x ** feat_power, span=s).mean() for s in fallback_spans}
                X_fb = pd.DataFrame.from_dict(ewma_fb, orient="columns")
                reg_fb = RidgeCV()
                reg_fb.fit(X_fb, y ** feat_power)
                coef = reg_fb.coef_
                spans_used = fallback_spans
                lambdas = 2 / (1 + spans_used)
            elif self.use_lasso:
                mask = coef != 0
                spans_used = self.spans[mask]
                coef = coef[mask]
                lambdas = lambdas[mask]
            else:
                spans_used = self.spans

            print(f"spans chosen: {spans_used}")

            timestamps = np.arange(max(spans_used))
            dts = timestamps * dt
            exp_kernel = (coef * lambdas * (1 - lambdas) ** timestamps.reshape(-1, 1)).sum(axis=1)
            exp_kernel /= exp_kernel.sum()

            try:
                opt_coef, _ = curve_fit(self.Kernel.kernel_with_coef, dts, exp_kernel, maxfev=4000)
            except RuntimeError:
                opt_coef = np.array([1, 1, 10 * dt])

            if self.plot:
                pred_kernel = self.Kernel.kernel_with_coef(dts, *opt_coef)
                plt.plot(dts, pred_kernel, label="best_fit", linestyle="--")
                plt.plot(dts, exp_kernel, label="exp_kernel", alpha=0.5)
                plt.legend()
                plt.show()

            init_alphas.append(opt_coef[1])
            init_deltas.append(opt_coef[2])


        features = self.comp_features(X, init_alphas, init_deltas)
        design_X = np.column_stack(list(features.values()))

        if self.n_extra:
            extra_arrays = []
            if self.neg_ret_feat:
                neg_ret = pd.Series(np.minimum(X.iloc[:, 1].values, 0), index=X.index)
                extra_arrays.append(neg_ret)
            if self.pos_ret_feat:
                pos_ret = pd.Series(np.maximum(X.iloc[:, 1].values, 0), index=X.index)
                extra_arrays.append(pos_ret)
            design_X = np.column_stack([design_X] + extra_arrays)

        reg = LinearRegression()
        reg.fit(design_X, y)

        init = np.concatenate([[reg.intercept_], reg.coef_[:self.num_feat], init_alphas, init_deltas])
        if self.n_extra:
            init = np.append(init, reg.coef_[-self.n_extra:])
        return init
    
    def fit(self, X_train, X_test, y_train, y_test):

        def residuals(params, X, y):
            return -y + self.lin_of_features(X, params)
        
        train_residuals = lambda params: residuals(params, X_train, y_train)

        def jacobian(params, X, y):

            if self.neg_ret_feat:
                core_params, beta_neg = params[:-1], params[-1]
            else:
                core_params = params
            
            features, pred = self.lin_of_features(X, params, ret_feats=True)
            intercept, betas, alphas, deltas = split_params(core_params, self.model_spec)

            jacob = np.zeros((len(params), len(y)))
            jacob[0] = 1 # intercept derivs

            for i, feat in enumerate(self.model_spec):
                sign, feat_power = parse_feat(feat)
                clipped_X = clip_returns(X, sign)

                R = features[feat]**feat_power
                dy_dbeta = features[feat]

                dR_dalpha = comp_weighted_sum(clipped_X, self.Kernel.dk_dalpha, alphas[i], deltas[i], feat_power)
                dy_dalpha = betas[i] / feat_power * np.abs(R)**(1/feat_power-1) * dR_dalpha

                dR_ddelta = comp_weighted_sum(clipped_X, self.Kernel.dk_ddelta, alphas[i], deltas[i], feat_power)
                dy_ddelta = betas[i] / feat_power * np.abs(R)**(1/feat_power-1) * dR_ddelta

                jacob[1+i] = dy_dbeta
                jacob[self.num_feat+1+i] = dy_dalpha
                jacob[2*self.num_feat+1+i] = dy_ddelta

            if self.n_extra:
                row = len(params) - self.n_extra
                if self.neg_ret_feat:
                    jacob[row] = np.minimum(X.iloc[:, 1].values, 0)
                    row += 1
                if self.pos_ret_feat:
                    jacob[row] = np.maximum(X.iloc[:, 1].values, 0)

            return jacob.T
                
        train_jacobian = lambda params: jacobian(params, X_train, y_train)

        lower_bound, upper_bound = self.Kernel.bounds(self.num_feat)

        if self.n_extra:
            lower_bound = np.append(lower_bound, [-np.inf for _ in range(self.n_extra)])
            upper_bound = np.append(upper_bound, [np.inf for _ in range(self.n_extra)])

        initial_params_train = self.init_params(X_train, y_train)
        initial_params_train = np.clip(initial_params_train, lower_bound, upper_bound)

        initial_pred_train = self.lin_of_features(X_train, initial_params_train)
        initial_pred_test = self.lin_of_features(X_test, initial_params_train)
        initial_pred_train = np.clip(initial_pred_train, 0, None)
        initial_pred_test = np.clip(initial_pred_test, 0, None)

        sol = least_squares(
            train_residuals,
            initial_params_train,
            method="trf",
            bounds=(lower_bound, upper_bound),
            jac=train_jacobian
        )
        ls_params = sol["x"]
        if self.n_extra:
            core_params = ls_params[:-self.n_extra]
            extra_betas = ls_params[-self.n_extra:]
        else:
            core_params = ls_params
        intercept, betas, alphas, deltas = split_params(core_params, self.model_spec)

        norm_consts = []
        for i, feat in enumerate(self.model_spec):
            _, feat_power = parse_feat(feat)
            norm_const = self.Kernel.norm_const(alphas[i], deltas[i], self.max_delta, 1/feat_power)
            norm_consts.append(norm_const)
        norm_consts = np.array(norm_consts)

        train_features, pred_train = self.lin_of_features(X_train, ls_params, ret_feats=True)
        test_features, pred_test = self.lin_of_features(X_test, ls_params, ret_feats=True)
        pred_train = np.clip(pred_train, 0, None)
        pred_test = np.clip(pred_test, 0, None)

        # normalise features and coefs
        for i, key in enumerate(train_features):
            train_features[key] /= norm_consts[i]
        for i, key in enumerate(test_features):
            test_features[key] /= norm_consts[i]
        betas *= norm_consts

        # features to df
        features_df = pd.DataFrame({
            str(key): pd.concat([train_features[key], test_features[key]])
            for key in train_features
        }).sort_index()

        result = {
            "sol": sol,
            "intercept": intercept, "betas": betas, "alphas": alphas, "deltas": deltas,
            "train_true": y_train, "test_true": y_test,
            "train_pred": pd.Series(pred_train, index=X_train.index), "test_pred": pd.Series(pred_test, index=X_test.index),
            "train_rmse": root_mean_squared_error(y_true=y_train, y_pred=pred_train),
            "test_rmse": root_mean_squared_error(y_true=y_test, y_pred=pred_test),
            "train_r2": r2_score(y_true=y_train, y_pred=pred_train),
            "test_r2": r2_score(y_true=y_test, y_pred=pred_test),
            "features": features_df,
            "initial_params": initial_params_train,
            "initial_train_rmse": root_mean_squared_error(y_true=y_train, y_pred=initial_pred_train),
            "initial_test_rmse": root_mean_squared_error(y_true=y_test, y_pred=initial_pred_test),
            "initial_train_r2": r2_score(y_true=y_train, y_pred=initial_pred_train),
            "initial_test_r2": r2_score(y_true=y_test, y_pred=initial_pred_test),
        }

        if self.n_extra:
            result["extra_betas"] = extra_betas

        return result
    
    def fit_2exp(self, alphas, deltas, fit_period=126):

        def exp2(tau, lambda0, lambda1, theta, c):
            exp_0 = lambda0 * np.exp(-lambda0 * tau)
            exp_1 = lambda1 * np.exp(-lambda1 * tau)
            return c * ((1-theta) * exp_0 + theta * exp_1)
        
        dts = np.arange(fit_period) * dt
        results = {}

        for i, feat in enumerate(self.model_spec):
            tspl_kernel = self.Kernel.kernel(dts, alphas[i], deltas[i])

            # bounds: lambda0, lambda1 >= 0, theta in [0,1], c >= 0
            lower_bound = np.array([0, 0, 0, 0])
            upper_bound = np.array([np.inf, np.inf, 1, np.inf])

            try:
                opt_params, _ = curve_fit(
                    exp2, dts, tspl_kernel,
                    bounds=(lower_bound, upper_bound), maxfev=4000
                )
            except RuntimeError:
                opt_params = np.array([60, 60, 0.5, 1])

            # convention: lam0 >= lam1 (swap if needed)
            lambda0, lambda1, theta, c = opt_params
            if lambda0 < lambda1:
                lambda0, lambda1, theta = lambda1, lambda0, 1 - theta

            results[feat] = {
                "lambda0": lambda0, "lambda1": lambda1,
                "theta": theta, "c": c
            }

        return results

def empirical_study(
        load_from, train_start_date, test_start_date, test_end_date,
        index, vol, index_suffix, vol_suffix,
        max_delta, KernelClass, model_spec,
        spans, cv_splits, use_lasso, plot,
        neg_ret_feat=False, pos_ret_feat=False, forecast=0,
        source="eodhd", api_key=None, filepath=None
):
    
    # data loading and prep
    X_train, X_test, y_train, y_test = get_data(
    index, vol, index_suffix, vol_suffix,
    load_from, train_start_date, test_start_date, test_end_date,
    max_delta=max_delta, forecast=forecast,
    source=source, api_key=api_key, filepath=filepath
    )

    # model
    PDV = PDVModel(
        KernelClass=KernelClass, model_spec=model_spec, max_delta=max_delta,
        spans=spans, cv_splits=cv_splits, use_lasso=use_lasso, plot=plot,
        neg_ret_feat=neg_ret_feat, pos_ret_feat=pos_ret_feat
        )

    results = PDV.fit(X_train, X_test, y_train, y_test)

    params_str = (
    f'beta_0/intercept: {results["intercept"]}, '\
    f'betas: {results["betas"]}, '\
    f'alphas: {results["alphas"]}, '\
    f'deltas: {results["deltas"]}'
    )

    if neg_ret_feat or pos_ret_feat:
        params_str += f", extra betas: {results["extra_betas"]}"

    print(params_str)

    print(
    f'train RMSE: {results["train_rmse"]}, '\
    f'train R-squared: {results["train_r2"]}, '\
    f'test RMSE: {results["test_rmse"]}, '\
    f'test R-squared: {results["test_r2"]}'
    )

    return results

def fit_arch_model(
        index, vol, index_suffix, vol_suffix,
        load_from, train_start_date, test_start_date, test_end_date,
        model_class=LinearRegression, max_delta=200, step_delta=5,values_of_p=None,
        forecast=0, source="eodhd", api_key=None, filepath=None
        ):
    
    # data loading and prep
    X_train, X_test, y_train, y_test = get_data(
    index, vol, index_suffix, vol_suffix,
    load_from, train_start_date, test_start_date, test_end_date,
    max_delta=max_delta, forecast=forecast,
    source=source, api_key=api_key, filepath=filepath
    )

    if values_of_p is None:
        values_of_p = np.arange(0, max_delta, step_delta)
        values_of_p[0] = 1

    scores = np.zeros((len(values_of_p)))
    for j, p in enumerate(values_of_p):
        # print(f'\r {j}', end=' ')
        x = X_train.iloc[:, :p] ** 2
        y = y_train ** 2
        model = model_class()
        model.fit(x, y)

        y_pred = model.predict(x)
        mse = root_mean_squared_error(y, y_pred)
        scores[j] = mse
    idx_p = np.argmin(scores)
    best_p = values_of_p[idx_p]

    model = model_class()
    x = X_train.iloc[:, :best_p] ** 2
    x_test = X_test.iloc[:, :best_p] ** 2
    y = y_train ** 2

    model.fit(x, y)
    test_pred = model.predict(x_test)
    # y_test = np.sqrt(y_test)
    test_pred = np.sqrt(np.clip(test_pred, 0, None))
    test_r2 = r2_score(y_true=y_test, y_pred=test_pred)
    test_rmse = root_mean_squared_error(y_true=y_test, y_pred=test_pred)

    train_pred = model.predict(x)
    train_pred = np.sqrt(np.clip(train_pred, 0, None))

    train_rmse = root_mean_squared_error(y_true=y_train, y_pred=train_pred)
    train_r2 = r2_score(y_true=y_train, y_pred=train_pred)
    output = {'model': model,
              'best_p': best_p,
              'opt_params': {'beta_0': model.intercept_, 'betas': model.coef_},
              # 'validation_r2': 1 - scores[idx_p][1].mean() / np.var(np.sqrt(y)),
              # 'mean_r2_validation': mean_r2_per_p[idx_p],
              # 'median_r2_validation': median_r2_per_p[idx_p],
              # 'validation_rmse': np.sqrt(scores[idx_p][1].mean()),
              'train_rmse': train_rmse,
              'train_r2': train_r2,
              'test_rmse': test_rmse,
              'test_r2': test_r2,
              'test_pred': pd.Series(test_pred, index=y_test.index),
              'train_pred': pd.Series(train_pred, index=y_train.index),
              'scores': scores
              }
    return output