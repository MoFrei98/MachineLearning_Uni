from ml.models.svm_model import SVMModel
from ml.models.tree_model import TreeModel
from ml.models.knn_model import KNNModel
from ml.models.randomforest_model import RandomForestModel
from ml.models.lda_model import LDAModel
from ml.models.gxboost_model import XGBoostModel
from ml.models.model import Model
from ml.kernel.linear_kernel import LinearKernel
from ml.kernel.rbf_kernel import RBFKernel
from ml.kernel.poly_kernel import PolyKernel
from ml.kernel.sigmoid_kernel import SigmoidKernel
from typing import Type, Tuple, Dict, Any

class ModelFactory:
    _registry: Dict[str, Tuple[Type[Model], Dict[str, Any]]] = {
        # --- Strategy 1-4: SVM Variants with dedicated Kernel classes ---
        "svm_linear":    (SVMModel,         {"kernel": LinearKernel()}),
        "svm_rbf":       (SVMModel,         {"kernel": RBFKernel()}),
        "svm_poly":      (SVMModel,         {"kernel": PolyKernel(degree=3)}),
        "svm_sigmoid":   (SVMModel,         {"kernel": SigmoidKernel()}),
        # --- Strategy 5: Decision Tree ---
        "decision_tree": (TreeModel,        {"max_depth": 5}),
        # --- Strategy 6: K-Nearest Neighbors ---
        "knn":           (KNNModel,         {"n_neighbors": 5}),
        # --- Strategy 7: Random Forest ---
        "random_forest": (RandomForestModel,{"n_estimators": 100}),
        # --- Strategy 8: Linear Discriminant Analysis ---
        "lda":           (LDAModel,         {}),
        # --- Strategy 9: XGBoost ---
        "xgboost":       (XGBoostModel,     {"n_estimators": 100, "random_state": 42}),
    }

    @staticmethod
    def get_model(algo_type: str, input_shape: int, output_shape: int) -> Model:
        if algo_type not in ModelFactory._registry:
            raise ValueError(f"Unknown algorithm type: {algo_type}")

        model_class, build_params = ModelFactory._registry[algo_type]
        model: Model = model_class(input_shape, output_shape)
        model.build(**build_params)
        return model