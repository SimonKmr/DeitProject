class Validation:
    def __init__(self,avr_loss,top_1_acc,top_5_acc):
        self.top_1_acc = top_1_acc
        self.top_5_acc = top_5_acc
        self.avr_loss = avr_loss
        return