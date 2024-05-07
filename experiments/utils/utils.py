# https://stackoverflow.com/a/33879038/10949124
def partitionIndexes(totalsize, numberofpartitions):
    # Compute the chunk size (integer division; i.e. assuming Python 2.7)
    chunksize = totalsize // numberofpartitions
    # How many chunks need an extra 1 added to the size?
    remainder = totalsize - chunksize * numberofpartitions
    a = 0
    for i in range(numberofpartitions):
        b = a + chunksize + (i < remainder)
        # Yield the inclusive-inclusive range
        yield a
        a = b
