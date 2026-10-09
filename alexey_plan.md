1. пайплайн и структура есть
2. шаблон кросс-валидации уже есть
3. взять шаблон и составить для новых вариантов


## применить новые варианты кросс-валидации

### StratifiedKFold
K-Fold то есть данные делятся без перемешивания на K разных частей, но каждый фолд сохраняет распределение целевой переменной (обычно для классификации, но можно дискретизировать target).

###  GroupKFold
K-Fold но объекты из одной группы всегда в одном фолде

###  GroupShuffleSplit

Grouped K-Fold with Random Permutation. This is a combination of two methods:
- We define groups
- and also randomly sample the entire dataset during each iteration to generate a training set and a validation set.


###  LeaveOneGroupOut

In each iteration the model is trained with samples of all groups but one. In the case of the months as groups, 12 iterations are performed.