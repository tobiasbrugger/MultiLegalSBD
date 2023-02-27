from itertools import chain

from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer


class CRF_Features():
    """
    Creates features for a CRF-Model
    """
    specialWindow = 10  # 10
    lowercaseWindow = 7  # 7
    lengthWindow = 7  # 7
    signWindow = 5  # 5
    lowerWindow = 3
    upperWindow = 3
    numberWindow = 3
    spaceWindow = 3

    featWindows = [specialWindow, lowercaseWindow, lengthWindow,
                   signWindow, lowerWindow, upperWindow, numberWindow, spaceWindow]

    def generate_features(self, tokens):
        features = self.sent2features(tokens)
        return features

    def isSpecial(self, token: str) -> str:
        """
        Checks wether a token is a special character or not and returns the stringified decisions 
        :param str token: the token to be checked
        :return: String
        :rtype: str
        """
        if len(token) > 1 or token.isalnum():
            return "No"
        elif token in ["'"]:
            return 'Abbr'
        elif token in ['.', '!', '?', ':']:
            return 'End'
        elif token in ['(', '[', '{']:
            return 'Open'
        elif token in [')', ']', '}']:
            return 'Close'
        elif token in ['\r', '\n']:
            return 'Newline'
        else:
            return 'S'

    def sign(self, token: str) -> str:
        """
        Converts a given token into a signature
        :param str token: the token to be converted
        :return: Signature of token
        :rtype: str
        """
        sig = []
        for c in token:
            if c.isnumeric():
                sig.append("N")
            elif c.islower():
                sig.append("c")
            elif c.isupper():
                sig.append("C")
            else:
                sig.append("S")

        return ''.join(sig)

    def firstLower(self, token):
        return token[0].islower()

    def firstUpper(self, token):
        return token[0].isupper()

    def token2features(self, sent, i) -> dict:
        """
        Calculates features for a given token in a sentence
        :param list sent: The sentence that contains the token
        :param int i: The index of the token to be calculated
        :return: Dictionary of features
        """
        if isinstance(sent[i], tuple):
            token = sent[i][0]
        else:
            token = sent[i]
        """
        Special -> isSpecial(token)
        Lowercase -> token.lower()
        Length -> len(token)
        Signature -> sign(token)
        Upper -> self.firstUpper(token)
        Lower -> self.firstLower(token)
        Number -> token.isdigit()
        Space -> token.isspace()
        """
        features = {
            'bias': 1.0,
            '0:lowercase': token.lower(),
            '0:lower': self.firstLower(token),
            '0:upper': self.firstUpper(token),
            '0:numeric': token.isnumeric(),
            '0:special': self.isSpecial(token),
            '0:sign': self.sign(token),
            '0:length': len(token),
            '0:BOS': False,
            '0:EOS': False
        }
        if i == 0:
            features.update({
                '0:BOS': True
            })
        elif i == len(sent)-1:
            features.update({
                '0:EOS': True
            })

        y = 0
        maxLeft = max([i, 0])
        while i > y and y <= max(self.featWindows) and y <= maxLeft:
            y = y+1

            if isinstance(sent[i], tuple):
                tempToken = sent[i-y][0]
            else:
                tempToken = sent[i-y]
            number = str((y)*-1)
            if y == maxLeft:
                features.update({
                    number+':BOS': True
                })
            else:
                features.update({
                    number+':BOS': False
                })
            if y <= self.specialWindow:
                features.update({
                    number+':special': self.isSpecial(tempToken)
                })
            if y <= self.lowercaseWindow:
                features.update({
                    number+':lowercase': tempToken.lower()
                })
            if y <= self.lengthWindow:
                features.update({
                    number+':length': len(tempToken)
                })
            if y <= self.signWindow:
                features.update({
                    number+':sign': self.sign(tempToken)
                })
            if y <= self.lowerWindow:
                features.update({
                    number+':lower': self.firstLower(tempToken)
                })
            if y <= self.upperWindow:
                features.update({
                    number+':upper': self.firstUpper(tempToken)
                })
            if y <= self.numberWindow:
                features.update({
                    number+':number': tempToken.isnumeric()
                })
            if y <= self.spaceWindow:
                features.update({
                    number+':space': tempToken.isspace()
                })

        y = 0
        maxRight = len(sent) - i-1
        while y <= max(self.featWindows) and y < maxRight:
            y = y + 1
            if isinstance(sent[i], tuple):
                tempToken = sent[i+y][0]
            else:
                tempToken = sent[i+y]
            number = '+'+str((y))
            if y == maxRight:
                features.update({
                    number+':EOS': True
                })
            else:
                features.update({
                    number+':EOS': False
                })
            if y <= self.specialWindow:
                features.update({
                    number+':special': self.isSpecial(tempToken)
                })
            if y <= self.lowercaseWindow:
                features.update({
                    number+':lowercase': tempToken.lower()
                })
            if y <= self.lengthWindow:
                features.update({
                    number+':length': len(tempToken)
                })
            if y <= self.signWindow:
                features.update({
                    number+':sign': self.sign(tempToken)
                })
            if y <= self.lowerWindow:
                features.update({
                    number+':lower': self.firstLower(tempToken)
                })
            if y <= self.upperWindow:
                features.update({
                    number+':upper': self.firstUpper(tempToken)
                })
            if y <= self.numberWindow:
                features.update({
                    number+':number': tempToken.isnumeric()
                })
            if y <= self.spaceWindow:
                features.update({
                    number+':space': tempToken.isspace()
                })
        return features

    def sent2features(self, sent):
        return [self.token2features(sent, i) for i in range(len(sent))]

    def sent2labels(self, sent):
        return [label for token, label in sent]

    def sent2tokens(self, sent):
        return [token for token, label in sent]

    def bio_classification_report(self, y_true, y_pred):

        lb = LabelBinarizer()
        y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
        y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

        tagset = set(lb.classes_)
        # tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
        class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

        return classification_report(
            y_true_combined,
            y_pred_combined,
            labels=[class_indices[cls] for cls in tagset],
            target_names=tagset,
        )
