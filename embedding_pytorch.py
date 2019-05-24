import torch
from torch.autograd import Variable

class MatrixFactorization(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors=20):
        super().__init__()
	# create user embeddings
        self.user_factors = torch.nn.Embedding(n_users, n_factors,
                                               sparse=False)
	# create item embeddings
        self.item_factors = torch.nn.Embedding(n_items, n_factors,
                                               sparse=False)
        self.item_bias = torch.nn.Embedding(n_items, 1,
                                               sparse=False)
        self.user_bias = torch.nn.Embedding(n_users, 1, sparse=False)




    def forward(self, user, item):
    	# matrix multiplication + item bias + user bias

        self.add1 = (self.user_factors(user)*self.item_factors(item)).sum(1).add( self.user_bias(user) ).add(  self.item_bias(item) )
        return torch.nn.Sigmoid(self.add1)
        # return torch.add( torch.add( (self.user_factors(user)*self.item_factors(item)).sum(1), self.user_bias(user)), self.item_bias(item) )

    def predict(self, user, item):
        return self.forward(user, item)


data_name = 'Assistment09'
item = 'problem'
dataSet = AssistmentData(name=data_name, item=item)
data = dataSet.data.values
n_users = dataSet.user_num
n_items = dataSet.item_num
model = MatrixFactorization(n_users, n_items, n_factors=20)
loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(),
                            lr=1e-6)

train, validation = train_test_split(data, test_size = 0.2)

users = train[:,0]
items = train[:,1]
ratings = train[:,2]

for user, item, rating in zip(users, items, ratings):
    # get user, item and rating data
    rating = Variable(torch.FloatTensor(int(rating)))
    user = Variable(torch.LongTensor([int(user)]))
    item = Variable(torch.LongTensor([int(item)]))

    # predict
    prediction = model(user, item)
    loss = loss_fn(prediction, rating)

    # backpropagate
    loss.backward()

    # update weights
    optimizer.step()
