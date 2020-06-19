
class MLpredictor():
    """This class serves as the basis for constructing predictors"""
    
    def __init__(self, name):
        self.name = name
        self.model = None


    def train(self):
        pass


    def save_model(self):
        # Construct the path to save to
        pass


    def load_model(self):
        pass


    def predict(self):
        """Makes a single prediction"""
        pass


    def build(self):
        """Build a full feature"""
        pass


if __name__ == "__main__":

    p = MLpredictor('knn')
    