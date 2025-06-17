class Stats:
    def __init__(self,avr_loss,top_1_acc,top_5_acc,  f1_score, precision, recall ):
        self.epoch = -1
        self.seconds = -1

        self.avr_loss = avr_loss
        self.top_1_acc = top_1_acc
        self.top_5_acc = top_5_acc

        self.f1_score = f1_score
        self.precision = precision
        self.recall = recall
        return

    def csv(self):
        return f"{self.avr_loss};{self.top_1_acc};{self.top_5_acc};{self.f1_score};{self.precision};{self.recall}\n"

    def __repr__(self):
        return f"avr_loss: {self.avr_loss:.4}, acc1: {self.top_1_acc:.4}, acc5: {self.top_5_acc:.4}, f1: {self.f1_score:.4}, prec: {self.precision:.4}, rcal: {self.recall:.4}"

    __str__ = __repr__