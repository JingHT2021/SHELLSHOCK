"""Round-robin finite iterators prevent one route monopolizing the budget."""
def _round_robin(iterators):
    active=list(map(iter,iterators))
    while active:
        remaining=[]
        for iterator in active:
            try:
                yield next(iterator)
                remaining.append(iterator)
            except StopIteration:
                pass
        active=remaining

