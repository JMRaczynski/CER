from module import PETER, MLP


class CER(PETER):
    def __init__(self, *args, **kwargs):
        super(CER, self).__init__(*args, **kwargs)
        self.original_recommender = MLP(self.emsize)
        self.additional_recommender = MLP(self.emsize)

    def predict_rating(self, hidden):
        rating = self.original_recommender(hidden[0])  # (batch_size,)
        hidden_explanation, _ = hidden[self.src_len:].max(0)
        explanation_rating = self.additional_recommender(hidden_explanation)
        return rating, explanation_rating
