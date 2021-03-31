import torch.nn as nn
import torch

""" for some reason pylint doesn't recognize torch's tensor manip functions, but they're thee."""

class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature):
        super(NT_Xent, self).__init__()

        # define some useful parameters for the loss function
        self.batch_size = batch_size
        self.temperature = temperature
        self.N = 2 * batch_size
        self.similarity_f = nn.CosineSimilarity(dim=2)


    # todo: vectorize and refactor to use nn.CEL
    def forward(self, z_i, z_j):

        # concatenate the representations of both transforms of the batch's images into
        # a single matrix, size [(N, 64)]
        reps = torch.cat([z_i, z_j], dim=0)

        # calculate temperature adjusted cosine similarity matrix and e^sim
        # exp(sim(u,v) / T). size [(N, N)]
        sim = self.similarity_f(reps.unsqueeze(1),reps.unsqueeze(0)) / self.temperature
        exp_sim = torch.exp(sim)

        # pull out positive pairs indexed by (i, batch_size+i) and (batch_size+j, j)
        # vectors of size [(batch_size)]
        sim_i_j = torch.diag(exp_sim, self.batch_size)
        sim_j_i = torch.diag(exp_sim, -self.batch_size)

        # pull out similarity to self, should be a vector of 1s, size [(N)]
        sim_self = torch.diag(exp_sim)

        pos_pairs = torch.cat([sim_i_j, sim_j_i], dim=0)

        # sum each row of the similarity matrix, subtracting out self-similarity. This gives
        # the denominator of the NT-Xent loss function. size [(N)]
        denominator = torch.sum(exp_sim, dim=1) - sim_self

        # divide positive pairs by sums of total similarities item-wise
        itemized_loss = -torch.log(torch.div(pos_pairs,denominator))

        # mean loss for the batch
        batch_loss = torch.mean(itemized_loss)

        return batch_loss