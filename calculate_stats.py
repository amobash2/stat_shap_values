import pandas as pd
import numpy as np
from scipy.spatial.distance import correlation
from scipy.stats import entropy
from typing import Dict, List
import catboost
import xgboost as xg
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import copy
import shap
import itertools
from scipy.stats import ttest_ind
from statsmodels.stats.outliers_influence import variance_inflation_factor

MIN_OBSERVATION_PER_CAT = 50
NUM_FOLDS = 10
RANDOM_STATE = 1234

class CalStatistics:
    def __init__ (self,
                  input_data: pd.DataFrame(),
                  target: str,
                  features: List[str],
                  categorical_features: List[str] = None,
                  calculate_distance_corr: bool = True,
                  test_size: float = 0.1,
                  num_imp_features: int = 50):
        self._input_data = input_data[features + [target]]
        self._input_data.fillna(0, inplace = True)
        self._input_data[features + [target]] = self._input_data[features + [target]].astype(float)
        self._target = target
        self._features = features
        self._categorical_features = categorical_features
        self._test_size = test_size
        self._num_imp_features = num_imp_features
        print("Calculating correlations...")
        self._corr, self._dist_corr = self._cal_corr(calculate_distance_corr)
        print("Calculating entropies...")
        self._entropy_df = self._cal_entropy()
        print("Calculating SHAP values...")
        self._tree_importance_df, \
             self._shap_df = self._cal_example_shap_values()
        print("Calculating VIF...")
        self._vif_df = self._cal_vif()
        print("Calculating test statistics for given categorical variables...")
        self._test_stats = self._cal_test_stats()
        print("Calculating overall data statistics...")
        self._data_level_stats = self._cal_data_level_statistics()

    @property
    def corr(self):
        self._corr.rename(columns = {"level_0": "SOURCE", "level_1": "TARGET"}, inplace=True)
        return self._corr
    
    @property
    def dist_corr(self):
        self._dist_corr.rename(columns = {"level_0": "SOURCE", "level_1": "TARGET"}, inplace=True)
        return self._dist_corr
    
    @property
    def entropy_df(self):
        return self._entropy_df
    
    @property
    def tree_importance_df(self):
        return self._tree_importance_df
    
    @property
    def shap_df(self):
        self._shap_df.rename(columns={"level_0": "OBS_NBR", "level_1": "FEATURE_NAME"}, inplace=True)
        self._shap_df["OBS_NBR"] = self._shap_df["OBS_NBR"] + 1
        return self._shap_df
    
    @property
    def vif_df(self):
        return self._vif_df
    
    @property
    def test_stats(self):
        return self._test_stats
    
    @property
    def data_level_stats(self):
        return self._data_level_stats

    def return_calculated_stats(self):
        return self.corr, \
            self.dist_corr, \
            self.entropy_df, \
            self.tree_importance_df, \
            self.shap_df, \
            self.vif_df, \
            self.test_stats, \
            self.data_level_stats

    def distcorr(self, x, y):
        return correlation(x,y)

    def _cal_corr(self, calculate_distance_corr: bool = True):
        try:
            print("Start calculating correlations...")
            corr = self._input_data.corr().stack().reset_index(name="CORRELATION")
            corr.fillna(0, inplace=True)
            print("Start calculating distance correlations...")
            dist_corr = self._input_data.corr(method=self.distcorr).stack().reset_index(name="DISTANCE_CORRELATION") if calculate_distance_corr else None
            dist_corr.fillna(0, inplace=True)
            return corr, dist_corr
        except:
            return None, None
        
    def _cal_entropy(self):
        entropy_values = []
        for c in list(self._input_data):
            try:
                entr_value = entropy(list(self._input_data[c]))
                if entr_value is not None and not pd.isnull(entr_value):
                    entropy_values.append({"FEATURE_NAME": c,
                                           "ENTROPY_VALUE": entr_value})
            except:
                print(f"Unable to calculate entropy for {c} feature!")
                pass

        entropy_df = pd.DataFrame(entropy_values)

        return entropy_df
    
    def _fit_catboost_model(self, x_train, x_test, y_train, y_test):

        train_dataset = catboost.Pool(x_train, y_train) 
        # test_dataset = catboost.Pool(x_test, y_test)

        model = catboost.CatBoostRegressor(loss_function='RMSE', random_state = RANDOM_STATE, verbose=False)
    
        grid = {'iterations': [200, 300, 400],
                'learning_rate': [0.03, 0.1, 0.01],
                'depth': [2, 4, 6, 8, 10],
                'l2_leaf_reg': [0.2, 0.5, 1, 3]} # This should be parameterized and fed from config file
        model.grid_search(grid, train_dataset, verbose = False)

        pred = model.predict(x_test)
        rmse = (np.sqrt(mean_squared_error(y_test, pred)))
        r2 = r2_score(y_test, pred)
        print("Testing performance")
        print("RMSE CatBoost: {:.2f}".format(rmse))
        print("R2 CatBoost: {:.2f}".format(r2))
        return model, pred
    
    def _fit_xgboost_model(self, x_train, x_test, y_train, y_test):
        model = xg.XGBRegressor(objective ='reg:squarederror', 
                                n_estimators = 100,
                                random_state = RANDOM_STATE,
                                verbose = False) 
        model.fit(x_train, y_train) 
        
        pred = model.predict(x_test) 
        rmse = (np.sqrt(mean_squared_error(y_test, pred)))
        r2 = r2_score(y_test, pred)
        print("Testing performance")
        print("RMSE XGBoost: {:.2f}".format(rmse))
        print("R2 XGBoost: {:.2f}".format(r2))
        return model, pred
    
    def _process_data_for_storage(self, value):
        try:
            if pd.isnull(value) or value is None:
                return None
            else:
                if isinstance(value, (int, float)):
                    return str(round(value, 3))
                elif isinstance(value, str):
                    return value
                else:
                    return str(value)
        except:
            return None
        
    def _preparing_feat_imp_tree(self, model, x_test):
        feature_importance = model.feature_importances_
        sorted_idx = np.argsort(feature_importance)
        tree_important_features = [d for d in x_test.columns[sorted_idx[-self._num_imp_features:]]]
        tree_importance_values = [round(v, 5) for v in feature_importance[sorted_idx[-self._num_imp_features:]]]
        return tree_important_features, tree_importance_values

    def _preparing_feat_imp_df(self, tree_important_features, tree_importance_values, model_idx):
        feat_imp_df = pd.DataFrame({"FEATURE_NAME": tree_important_features, "FEATURE_IMPORTANCE": tree_importance_values})
        feat_imp_df["MODEL_INDEX"] = model_idx + 1
        return feat_imp_df
    
    def _preparing_shap_df(self, shap_values, x_test, model_idx, model_name):
        shap_df = pd.DataFrame(shap_values, columns=[d for d in x_test.columns]).stack().reset_index(name="SHAP_VALUE")
        shap_df["MODEL_INDEX"] = model_idx + 1
        shap_df["MODEL_NAME"] = model_name
        return shap_df
    
    def _update_shap_df(self, shap_df, y_test, pred, x_test):
        feat_vals = []
        original_y = []
        predicted_y = []

        for _, row in shap_df.iterrows():
            try:
                original_y.append(self._process_data_for_storage(list(y_test)[row["level_0"]]))
            except:
                original_y.append(None)
            try:
                predicted_y.append(self._process_data_for_storage(list(pred)[row["level_0"]]))
            except:
                predicted_y.append(None)
            try:
                feat_vals.append(self._process_data_for_storage(list(x_test[row["level_1"]])[row["level_0"]]))
            except:
                feat_vals.append(None)
        shap_df["FEATURE_VALUE"] = feat_vals
        shap_df["TARGET_VALUE"] = original_y
        shap_df["PREDICTED_VALUE"] = predicted_y
        return shap_df
    
    def _preparing_shap_feat_imp_dfs(self, model, x_test, y_test, model_idx, pred, model_name: str):
        tree_important_features, tree_importance_values = self._preparing_feat_imp_tree(model, x_test)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_test)
        feat_imp_df = self._preparing_feat_imp_df(tree_important_features, tree_importance_values, model_idx)
        print("Finished processing all feature importances...")
        shap_df = self._preparing_shap_df(shap_values, x_test, model_idx, model_name)
        print("Finished processing all shap values...")
        shap_df = self._update_shap_df(shap_df, y_test, pred, x_test)
        print("Finished adding FEATURE_VALUE to shap_df...")
        return feat_imp_df, shap_df

    def _cal_example_shap_values(self):
        df = copy.deepcopy(self._input_data[self._input_data[self._target].notna()])
        x = df[list(set(self._features) - set([self._target]))]
        y = df[self._target]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= self._test_size)

        kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state = RANDOM_STATE)

        all_feats = []
        all_shaps = []

        print("Start of CatBoost training...")
        for i, (train_index, valid_index) in enumerate(kf.split(x_train)):
            print(f"Training for fold {i + 1}-th")
            temp_x_train, temp_y_train = x_train.iloc[train_index], y_train.iloc[train_index]
            temp_x_valid, temp_y_valid = x_train.iloc[valid_index], y_train.iloc[valid_index]
            print(f"Finished preparing training data for this fold of size = {len(temp_x_train)}")

            cat_model, cat_pred = self._fit_catboost_model(temp_x_train, temp_x_valid, temp_y_train, temp_y_valid)
            cat_feat_imp_df, cat_shap_df = self._preparing_shap_feat_imp_dfs(cat_model, x_test, y_test, i, cat_pred, model_name = "CatBoost")

            xg_model, xg_pred = self._fit_xgboost_model(temp_x_train, temp_x_valid, temp_y_train, temp_y_valid)
            xg_feat_imp_df, xg_shap_df = self._preparing_shap_feat_imp_dfs(xg_model, x_test, y_test, i + NUM_FOLDS, xg_pred, model_name = "XGBoost")

            all_feats.append(cat_feat_imp_df)
            all_shaps.append(cat_shap_df)
            all_feats.append(xg_feat_imp_df)
            all_shaps.append(xg_shap_df)

        print("Done with all feature importance and shap values...")
        return pd.concat(all_feats), \
               pd.concat(all_shaps)

    def _cal_vif(self):
        vif_df = pd.DataFrame()
        X = self._input_data.dropna()
        vif_df["FEATURE_NAME"] = [d for d in list(X)]
        vif_df["VIF"] = [variance_inflation_factor(X.values, i) 
                          for i in range(len(X.columns))] 
        return vif_df
    
    def _cal_test_stats(self):
        test_results = []
        if self._categorical_features is not None:
            for c in self._categorical_features:
                distinct_values = list(self._complete_data[c].drop_duplicates())
                if len(distinct_values) > 1:
                    hyp_data = []
                    for d in distinct_values:
                        if not pd.isnull(d):
                            sub_data = list(self._complete_data.loc[self._complete_data[c] == d][self._target])
                            if len(sub_data) > MIN_OBSERVATION_PER_CAT:
                                hyp_data.append({"category": c, "category_value": str(d), "data": sub_data})
                        else:
                            sub_data = list(self._complete_data.loc[pd.isnull(self._complete_data[c])][self._target])
                            if len(sub_data) > MIN_OBSERVATION_PER_CAT:
                                hyp_data.append({"category": c, "category_value": "Null", "data": sub_data})

                combs = list(itertools.combinations(hyp_data, 2))
                for f in combs:
                    stat, pvalue = ttest_ind(f[0]["data"], f[1]["data"])
                    num_obs0, mn0, sd0, mdn0, per750, per900 = self._cal_statistics(f[0]["data"])
                    num_obs1, mn1, sd1, mdn1, per751, per901 = self._cal_statistics(f[1]["data"])
                    test_results.append({"FIRST_VARIABLE": f[0]["category"],
                                        "FIRST_VARIABLE_VALUE": f[0]["category_value"],
                                        "SECOND_VARIABLE": f[1]["category"],
                                        "SECOND_VARIABLE_VALUE": f[1]["category_value"],
                                        "TARGET": self._target,
                                        "STATISTIC": stat, "P_VALUE": pvalue,
                                        "FIRST_VARIABLE_NUM_OBS": num_obs0,
                                        "FIRST_VARIABLE_MEAN": mn0,
                                        "FIRST_VARIABLE_STD": sd0,
                                        "FIRST_VARIABLE_MEDIAN": mdn0,
                                        "FIRST_VARIABLE_75TH_PERCENTILE": per750,
                                        "FIRST_VARIABLE_90TH_PERCENTILE": per900,
                                        "SECOND_VARIABLE_NUM_OBS": num_obs1,
                                        "SECOND_VARIABLE_MEAN": mn1,
                                        "SECOND_VARIABLE_STD": sd1,
                                        "SECOND_VARIABLE_MEDIAN": mdn1,
                                        "SECOND_VARIABLE_75TH_PERCENTILE": per751,
                                        "SECOND_VARIABLE_90TH_PERCENTILE": per901})
        return pd.DataFrame(test_results)
        
    def _cal_data_level_statistics(self):
        data_level_stats = []
        for var in self._features + [self._target]:
            num_obs, mn, sd, mdn, per75, per90 = 0, 0, 0, 0, 0, 0
            try:
                num_obs, mn, sd, mdn, per75, per90 = self._cal_statistics(list(filter(None, self._input_data[var])))
            except:
                print(f"Unable to calculate statistics for the variable {var}...")
            data_level_stats.append({"FEATURE_NAME": var,
                                    "NUMBER_OBS": num_obs,
                                    "MEAN": mn,
                                    "STD": sd,
                                    "MEDIAN": mdn,
                                    "PERCENTILE_75TH": per75,
                                    "PERCENTILE_90TH": per90})
        return pd.DataFrame(data_level_stats)
    
    def _cal_statistics(self, data):
        num_obs = len(data)
        mn = np.mean(data)
        sd = np.std(data)
        mdn = np.median(data)
        per75 = np.percentile(data, 75)
        per90 = np.percentile(data, 90)
        return num_obs, mn, sd, mdn, per75, per90