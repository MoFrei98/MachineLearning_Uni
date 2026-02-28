from ml.models.svm_model import SVMModel

# TODO: create separate kernal classes with a base class and use them in the models
class ModelFactory:
    @staticmethod
    def get_model(algo_type, input_shape, output_shape):
        match algo_type:
            # --- Strategie 1 bis 4: SVM Varianten ---
            case "svm_linear":
                model = SVMModel(input_shape, output_shape)
                model.build(kernel='linear')
                return model

            case "svm_rbf":
                model = SVMModel(input_shape, output_shape)
                model.build(kernel='rbf')
                return model

            case "svm_poly":
                model = SVMModel(input_shape, output_shape)
                model.build(kernel='poly', degree=3)  # Degree ist spezifisch für Poly
                return model

            case "svm_sigmoid":
                model = SVMModel(input_shape, output_shape)
                model.build(kernel='sigmoid')
                return model

            # --- Hier folgen später die restlichen 4 Strategien ---
            # case "decision_tree":
            #     ...

            case _:
                raise ValueError(f"Unknown algorithm type: {algo_type}")