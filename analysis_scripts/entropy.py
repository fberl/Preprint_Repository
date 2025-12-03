import numpy as np
def compute_entropy(monkey_actions, computer_actions = None,N=2, minlen = 3, maxlen = 8):
    all_entropies = []
    if computer_actions is None:
        for seqlen in range(minlen,maxlen+1):
            monkey_entropy = {}
            for i in range(len(monkey_actions) - seqlen):
                seq = tuple(monkey_actions[i:i+seqlen+1])

                if seq in monkey_entropy:
                    monkey_entropy[seq] += 1
                else:
                    monkey_entropy[seq] = 1
            total = sum(monkey_entropy.values())
            monkey_entropy = {k: v/total for k,v in monkey_entropy.items()}
            
            # H = -sum(p*log(p)) + (k-1)/(np.log(N+1) * len(monkey_entropy))
            H = -sum([p*np.log(p) for p in monkey_entropy.values()]) + (2**seqlen-1)/(np.log(N+1) * len(monkey_entropy))
            all_entropies.append(H)
    else:
        # intersperse the two sequences. want monkey to always be first
        interspersed = []
        for i in range(len(monkey_actions)):
            interspersed.append(monkey_actions[i])
            interspersed.append(computer_actions[i])
        for seqlen in range(minlen,maxlen+1):
            monkey_entropy = {}
            for i in range(len(interspersed)//2 - seqlen):
                seq = tuple(interspersed[2*i:2*i+seqlen+1]) # want it to be inclusive
                if seq in monkey_entropy:
                    monkey_entropy[seq] += 1
                else:
                    monkey_entropy[seq] = 1
    # count the number of times each sequence appears
            total = sum(monkey_entropy.values())
            monkey_entropy = {k: v/total for k,v in monkey_entropy.items()}
            
            # H = -sum(p*log(p)) + (k-1)/(np.log(N+1) * len(monkey_entropy))
            H = -sum([p*np.log(p) for p in monkey_entropy.values()]) + (2**seqlen-1)/(np.log(N+1) * len(monkey_entropy))
            all_entropies.append(H)
    return all_entropies
    
def compute_mutual_information(monkey_actions, computer_actions=None, N=2, minlen = 3, maxlen = 8):
    # very similar to the entropy calculation, but now we need to calculate the joint probability of each sequence
    # the mutual info is I = - sum_i sum_j p_ij log_2(p_ij/(p_i*p_j)) - (N^seqlen-1)(N-1)/(np.log(N+1) * len(monkey_entropy))
    # easiest way is to do the same as the entropy calculation, but then take the last element in the sequence and use that as the other
    
    all_mutual_info = []
    if computer_actions is None:
        for seqlen in range(minlen,maxlen+1):
            monkey_mutual_info = {}
            monkey_seq = {}
            monkey_seq_trunc = {}
            monkey_seq_end = {}
            for i in range(len(monkey_actions) - seqlen-1):
                seq = tuple(monkey_actions[i:i+seqlen+1])
                seq_trunc = tuple(monkey_actions[i:i+seqlen])
                seq_end = tuple(monkey_actions[i+seqlen])
                if seq in monkey_mutual_info:
                    monkey_mutual_info[seq] += 1
                    monkey_seq[seq] += 1
                else:
                    monkey_mutual_info[seq] = 1
                    monkey_seq[seq] = 1
                if seq_trunc in monkey_seq_trunc:
                    monkey_seq_trunc[seq_trunc] += 1
                else:
                    monkey_seq_trunc[seq_trunc] = 1
                if seq_end in monkey_seq_end:
                    monkey_seq_end[seq_end] += 1
                else:
                    monkey_seq_end[seq_end] = 1
            total = sum(monkey_mutual_info.values())
            monkey_mutual_info = {k: v/total for k,v in monkey_mutual_info.items()}
            total_monkey_seq = sum(monkey_seq.values())
            monkey_seq = {k: v/total_monkey_seq for k,v in monkey_seq.items()}
            total_monkey_seq_trunc = sum(monkey_seq_trunc.values())
            monkey_seq_trunc = {k: v/total_monkey_seq_trunc for k,v in monkey_seq_trunc.items()}
            total_monkey_seq_end = sum(monkey_seq_end.values())
            monkey_seq_end = {k: v/total_monkey_seq_end for k,v in monkey_seq_end.items()}
            # I = - sum_i sum_j p_ij log_2(p_ij/(p_i*p_j)) - (N^seqlen-1)(N-1)/(np.log(N+1) * len(monkey_entropy))
            # need to form array of all pairs pij p_i p_j
            I = 0
            # need to verify this is the same as the double sum, but i think so?
            for seq_pair in monkey_mutual_info.keys():
                p_ij = monkey_mutual_info[seq_pair]
                p_i = monkey_seq_trunc[seq_pair[:-1]]
                p_j = monkey_seq_end[seq_pair[-1]]
                I += -p_ij*np.log2(p_ij/(p_i*p_j))
            if len(monkey_mutual_info) > 0:
                I += -(N**seqlen-1)*(N-1)/(np.log(N+1) * len(monkey_mutual_info))

            all_mutual_info.append(I)
    else:
        interspersed = []
        for i in range(len(monkey_actions)):
            interspersed.append(monkey_actions[i])
            interspersed.append(computer_actions[i]) 
        for seqlen in range(minlen,maxlen+1):
            monkey_mutual_info = {}
            monkey_seq = {}
            monkey_seq_trunc = {}
            monkey_seq_end = {}
            for i in range(len(interspersed)//2 - seqlen-1): # when we 
                seq = tuple(interspersed[2*i:2*i+seqlen+1]) # want it to be inclusive
                seq_trunc = seq[:-1]
                seq_end = seq[-1]
                if seq in monkey_mutual_info:
                    monkey_mutual_info[seq] += 1
                    monkey_seq[seq] += 1
                else:
                    monkey_mutual_info[seq] = 1
                    monkey_seq[seq] = 1
                if seq_trunc in monkey_seq_trunc:
                    monkey_seq_trunc[seq_trunc] += 1
                else:
                    monkey_seq_trunc[seq_trunc] = 1
                if seq_end in monkey_seq_end:
                    monkey_seq_end[seq_end] += 1
                else:
                    monkey_seq_end[seq_end] = 1
            total = sum(monkey_mutual_info.values())
            monkey_mutual_info = {k: v/total for k,v in monkey_mutual_info.items()}
            total_monkey_seq = sum(monkey_seq.values())
            monkey_seq = {k: v/total_monkey_seq for k,v in monkey_seq.items()}
            total_monkey_seq_trunc = sum(monkey_seq_trunc.values())
            monkey_seq_trunc = {k: v/total_monkey_seq_trunc for k,v in monkey_seq_trunc.items()}
            total_monkey_seq_end = sum(monkey_seq_end.values())
            monkey_seq_end = {k: v/total_monkey_seq_end for k,v in monkey_seq_end.items()}
            # I = - sum_i sum_j p_ij log_2(p_ij/(p_i*p_j)) - (N^seqlen-1)(N-1)/(np.log(N+1) * len(monkey_entropy))
            # need to form array of all pairs pij p_i p_j
            I = 0
            # need to verify this is the same as the double sum, but i think so?
            for seq_pair in monkey_mutual_info.keys():
                p_ij = monkey_mutual_info[seq_pair]
                p_i = monkey_seq_trunc[seq_pair[:-1]]
                p_j = monkey_seq_end[seq_pair[-1]]
                # assert(-p_ij*np.log2(p_ij/(p_i*p_j)) >= 0) 
                I += p_ij*np.log2(p_ij/(p_i*p_j))
            if len(monkey_mutual_info) > 0:
                I += (N**seqlen-1)*(N-1)/(np.log(N+1) * len(monkey_mutual_info))
            all_mutual_info.append(I)
    return all_mutual_info