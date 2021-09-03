

def pickModelWithHighestAdjustedR2(models):
    modelWithHighestAdjustedR2 = None
    highestAdjustedR2 = 0

    for model in models:
        adjustedR2OfCurrentModel = model.adjustedR2()
        if adjustedR2OfCurrentModel > highestAdjustedR2:
            modelWithHighestAdjustedR2 = model
            highestAdjustedR2 = adjustedR2OfCurrentModel

    return modelWithHighestAdjustedR2
