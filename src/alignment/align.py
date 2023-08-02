from diversity.task2vec import Task2Vec 

def alginment_with_diversity_coefficient(dataset_target,
                                        dataset_source,
                                        get_mapped_batch_fn: callable, 
                                        probe_network: nn.Module,
                                        tokenizer = None,
                                        batch_size: int = 512,
                                        num_batches: int = 100, 
                                        seed = 0, 
                                        buffer_size: int = 500_000, 
                                        distance = 'cosine',
                          ) -> dict:
    """
    Alignment v1 - with the Diversity Coefficient
    
    Given two data sets, compute how aligned they are using probe network f_w by comparing batches across the data sets:
        alg1 = align(T, S, f_w) = Align_1(T, S, f_w) = E_{B_s ~ S, B_t ~ T} [1 - d(e_{B_s}, e_{B_t})] =  1 - div(T, S)
    where e_{D} is the Task2Vec (diagonal of FIM) embedding of a batch D, and d is cosine distance function.
    
    ref: https://arxiv.org/abs/2306.13840
    """
    results: dict = cross_diversity_coefficient(dataset_target, dataset_source, get_mapped_batch_fn, probe_network, tokenizer, batch_size, num_batches, seed, buffer_size, distance)
    results['align'] = 1 - results['div_coeff']
    results['align_ci'] = results['div_coeff_ci']
    return results


def alignment_task2vec(dataset_target,
                        dataset_source,
                        get_mapped_batch_fn: callable,
                        probe_network: nn.Module,
                        tokenizer = None,
                        batch_size: int = 1024,
                        seed = 0, 
                        buffer_size: int = 500_000, 
                        distance = 'cosine',
                        ) -> dict:
    """
    Alignment v2 - with Task2Vec

    Given two data sets, compute how aligned they are using probe network f_w 
        alg_2 = Align_2(T, S, f_w) = 1 - d(e_{D_S}, e_{D_T})
    by comparing embedding the entire dataset or a large batch. 
    """

    # - Compute embedding of target
    shuffled_dataset = dataset_target.shuffle(buffer_size=buffer_size, seed=seed)
    tokenized_batch = get_mapped_batch_fn(shuffled_dataset)
    embedding_target, loss_target = Task2Vec(probe_network).embed(tokenized_batch)

    # - Compute embedding of source
    shuffled_dataset = dataset_source.shuffle(buffer_size=buffer_size, seed=seed)
    tokenized_batch = get_mapped_batch_fn(shuffled_dataset)
    embedding_source, loss_source = Task2Vec(probe_network).embed(tokenized_batch)

    # - Compute alignment
    distance_matrix = task_similarity.pdist([embedding_target, embedding_source], distance=distance)
    align = 1 - distance_matrix[0, 1]
    align_ci = task_similarity.stats_of_distance_matrix(distance_matrix)[1]

    # - Compute results
    embmbeddings, losses = [], []
    losses.append({'loss_target': loss_target, 'loss_source': loss_source})
    embeddings.append({'embedding_target': embedding_target, 'embedding_source': embedding_source})

    # - Results
    results: dict = {'align': align, 'align_ci': align_ci,
                    'embeddings': embeddings,
                    'distance_matrix': distance_matrix,
                    'losses': losses,
                    "batch_size": batch_size}
    return results
    