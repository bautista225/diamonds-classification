##### Libraries:
# using Pkg; 
# Pkg.add("Flux")
# Pkg.add("Statistics")
# Pkg.add("ScikitLearn");
# Pkg.add("PrettyTables")

using Flux;
using Flux.Losses;
using Random;
using Random:seed!
using DelimitedFiles;
using Statistics;
using ScikitLearn;
using PrettyTables;

@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier
@sk_import neural_network: MLPClassifier;
@sk_import ensemble:VotingClassifier
@sk_import ensemble:StackingClassifier
@sk_import ensemble:BaggingClassifier
@sk_import ensemble:(AdaBoostClassifier, GradientBoostingClassifier)
@sk_import ensemble:RandomForestClassifier


##### Functions:

# One Hot Encoding:

function oneHotEncoding(feature::AbstractArray{<:Any,1},      ## == Dummy Variables 
        classes::AbstractArray{<:Any,1})
    @assert length(classes) >= 2 "The number of classes is 0 or 1, it must be greater than 1"
    @assert length(classes) == length(unique(classes)) "Some class is repeated in the classes Array"
    if  length(classes) == 2
        enc_feature = reshape((classes[1] .== feature), (:,1))
    elseif length(classes) > 2
        enc_feature = convert(BitArray{2}, (classes .== permutedims(feature))')
    end
    return enc_feature
end;

oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature,unique(feature));

oneHotEncoding(feature::AbstractArray{Bool,1}) = reshape(feature, :, 1);

# Normalization:

function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    mins = minimum(dataset, dims=1)
    maxs = maximum(dataset, dims=1)
    return (mins,maxs)
end;

function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
    means = mean(dataset, dims=1)
    stds = std(dataset, dims=1)
    return (means,stds)
end;

function normalizeMinMax!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    @assert typeof(dataset[1]) != Int "Int elements are not mutable. Use Float elements instead"

    dataset[:,:] = ((dataset .- normalizationParameters[1]) ./ (normalizationParameters[2] - normalizationParameters[1]))
    dataset[:, vec(normalizationParameters[1] .== normalizationParameters[2])] .= 0   # ¿min=max? -> 0
    return dataset
end;

normalizeMinMax!(dataset::AbstractArray{<:Real,2}) = normalizeMinMax!(dataset, calculateMinMaxNormalizationParameters(dataset));

function normalizeMinMax(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}) 
    z_dataset = copy(dataset)
    z_dataset = normalizeMinMax!(z_dataset, normalizationParameters)
    return z_dataset
end;

normalizeMinMax(dataset::AbstractArray{<:Real,2}) = normalizeMinMax(dataset, calculateMinMaxNormalizationParameters(dataset));

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    @assert typeof(dataset[1]) != Int "Int elements are not mutable. Use Float elements instead"

    dataset[:,:] = (dataset .- normalizationParameters[1]) ./ normalizationParameters[2] # We standardize
    dataset[:, vec(normalizationParameters[2] .== 0)] .= 0 # We change the columns of the variables that have zero variability to columns with zero values      
    return dataset
end;

normalizeZeroMean!(dataset::AbstractArray{<:Real,2}) = normalizeZeroMean!(dataset, calculateZeroMeanNormalizationParameters(dataset));

function normalizeZeroMean(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    z_dataset = copy(dataset)
    z_dataset = normalizeZeroMean!(z_dataset, normalizationParameters)
    return z_dataset
end;

normalizeZeroMean(dataset::AbstractArray{<:Real,2}) = normalizeZeroMean(dataset, calculateZeroMeanNormalizationParameters(dataset));

# Classify Outputs:

function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5) 
    if size(outputs,2) == 1
        return outputs .>= threshold
    elseif size(outputs,2) > 1
        (_,indicesMaxEachInstance) = findmax(outputs, dims=2);
        outputs = falses(size(outputs));
        outputs[indicesMaxEachInstance] .= true;
        return outputs
    end
end;

# Accuracy:

function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    @assert length(outputs) == length(targets) "The size of the Arrays does not match"
    
    return mean(outputs .== targets)
end;

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    @assert size(outputs) == size(targets) "The size of the Arrays does not match"
    
    if ((size(targets,2) == 1) | (size(targets,2) == 2))
        return accuracy(outputs[:,1],targets[:,1])
    elseif size(targets,2) > 2
        return mean(all((outputs .== targets), dims=2))
    end
end;

function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    outputs = outputs .>= threshold
    return accuracy(outputs, targets)
end;

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5)
    @assert size(outputs) == size(targets) "The size of the Arrays does not match"

    if (size(targets,2) == 1) | (size(targets,2) == 2)
        return accuracy(outputs[:,1], targets[:,1], threshold=threshold)
    elseif size(targets,2) > 2
        outputs = classifyOutputs(outputs) # The value of threshold is not necessary (n_classes > 2)
        return accuracy(outputs, targets)
    end
end;

# Build ANN:

function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int;
                        transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))
    @assert length(topology) == length(transferFunctions) "Number of layers does not match with the number of activation functions"
    @assert numInputs >= 1 "Number of inputs must be greater than 0"
    @assert numOutputs >= 1 "Number of outputs must be greater than 0"

    ann = Chain(); # Create an empty ANN
    numInputsLayer = numInputs

    #Add hidden layers (if topology is not empty)
    if length(topology) != 0 
        layerIndex = 1
        for numOutputsLayer = topology      
            ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, transferFunctions[layerIndex]));      
            numInputsLayer = numOutputsLayer;
            layerIndex += 1 
        end;
    end

    #Add final layer (depending on the number of outputs / type of problem)
    if (numOutputs == 1) | (numOutputs == 2)
        ann = Chain(ann..., Dense(numInputsLayer, 1, σ)); #Final layer (numOutputs=1 | numOutputs=2 -> Sigmoid)
    else 
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity)); #Final layer (numOutputs>2 -> Softmax)
        ann = Chain(ann..., softmax) #Softmax 'layer'
    end

    return ann #Return the ANN
end;

# Train ANN:

function trainClassANN(topology::AbstractArray{<:Int,1},      
                        dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
                        transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
                        maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)
    @assert size(dataset[1],1) == size(dataset[2],1) "The number of patterns does not match in inputs and targets"
    @assert maxEpochs >= 1 "The maximum number of epochs has to be at least 1"

    ann = buildClassANN(size(dataset[1],2), topology, size(dataset[2],2), transferFunctions=transferFunctions)
    loss(x, y) = (size(dataset[2],2) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);
    loss_values = Vector{Float32}()
    accuracy_values = Vector{Float32}()

    for epoch in 1:maxEpochs
        Flux.train!(loss, Flux.params(ann), [(dataset[1]', dataset[2]')], ADAM(learningRate)) # Network training
        loss_value = loss(dataset[1]',dataset[2]')
        push!(loss_values, loss_value) #Save epoch loss value
        push!(accuracy_values, accuracy(ann(dataset[1]')', dataset[2])) #Save epoch accuracy value
        if loss_value <= minLoss #minLoss stopping criteria
            println("Value of minLoss achieved. Training process interrupted on the ", epoch, " epoch.")
            break
        end
    end

    return (ann, loss_values, accuracy_values) #Return the trained ANN, the loss values and the accuracy values.
end;

function trainClassANN(topology::AbstractArray{<:Int,1},      
                        (inputs, targets)::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};      
                        transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),      
                        maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)
    targets = reshape(targets, (:,1))
    return trainClassANN(topology, (inputs,targets), 
                            transferFunctions=transferFunctions, 
                            maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate)
end;

# Hold Out functions:

function holdOut(N::Int, P::Real)
    @assert (((P>=0.) & (P<=1.))) "P must take a value between 0 and 1"
    index = randperm(N)
    numTrainingSamples = Int(round(N*(1-P)))
    return (index[1:numTrainingSamples], index[numTrainingSamples+1:end])
end;

function holdOut(N::Int, Pval::Real, Ptest::Real) 
    @assert (((Pval>=0) & (Pval<=1))) "Pval must take a value between 0 and 1"
    @assert (((Ptest>=0) & (Ptest<=1))) "Ptest must take a value between 0 and 1"
    @assert (Pval + Ptest) <= 1 "Pval + Ptest must add up to less than one" 
    
    # Separate dataset into (training+validation) and test
    (trainValIndex,testIndex) = holdOut(N,Ptest)
    
    # Separate training and validation
    (trainIndex,valIndex) = holdOut(N-length(testIndex),Pval/(1-Ptest))
    
    trainIndex = trainValIndex[trainIndex]
    valIndex = trainValIndex[valIndex]
    
    return (trainIndex,valIndex,testIndex)
end;

# Loss Values:

function calculateLossValues(loss::Function,
                                trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; 
                                validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}= 
                                        (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)), 
                                testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}= 
                                        (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)))

    lossValValues = NaN
    lossTestValues = NaN

    if !isempty(validationDataset[1]) && !isempty(testDataset[1])
        lossTrainValues = loss(trainingDataset[1]',trainingDataset[2]')
        lossValValues = loss(validationDataset[1]',validationDataset[2]')
        lossTestValues = loss(testDataset[1]',testDataset[2]')
    elseif !isempty(testDataset[1])
        lossTrainValues = loss(trainingDataset[1]',trainingDataset[2]')
        lossTestValues = loss(testDataset[1]',testDataset[2]')
    elseif !isempty(validationDataset[1])
        lossTrainValues = loss(trainingDataset[1]',trainingDataset[2]')
        lossValValues = loss(validationDataset[1]',validationDataset[2]')
    else
        lossTrainValues = loss(trainingDataset[1]',trainingDataset[2]')
    end

    return (lossTrainValues,lossValValues,lossTestValues)
end;

# TrainANN (Allows validation set)

function trainClassANN(topology::AbstractArray{<:Int,1},  
                        trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; 
                        validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}= 
                                (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)), 
                        testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}= 
                                (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)), 
                        transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
                        maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,  
                        maxEpochsVal::Int=20, showText::Bool=false)

    @assert maxEpochs >= 1 "The maximum number of epochs has to be at least 1"
    @assert maxEpochsVal >= 1 "The maximum number of validation epochs has to be at least 1"

    (trainingInputs, trainingTargets) = trainingDataset;
    (validationInputs, validationTargets) = validationDataset;
    (testInputs, testTargets) = testDataset;

    @assert(size(trainingInputs, 1)==size(trainingTargets, 1)) 
        "The number of patterns in the training sets do not match"
    @assert(size(validationInputs, 1)==size(validationTargets, 1)) 
        "The number of patterns in the validation sets do not match"
    @assert(size(testInputs, 1)==size(testTargets, 1)) 
        "The number of patterns in the test sets do not match"

    #The number of columns in the training and validation inputs do not match:
    !isempty(validationInputs) && @assert(size(trainingInputs, 2)==size(validationInputs, 2));
    #The number of columns in the training and test inputs do not match:
    !isempty(testInputs) && @assert(size(trainingInputs, 2)==size(testInputs, 2));
    #The number of columns in the training and validation targets do not match:
    !isempty(validationTargets) && @assert(size(trainingTargets, 2)==size(validationTargets, 2));
    #The number of columns in the training and test targets do not match:
    !isempty(testTargets) && @assert(size(trainingTargets, 2)==size(testTargets, 2));

    #Define network
    ann = buildClassANN(size(trainingInputs,2), topology, size(trainingTargets,2), transferFunctions=transferFunctions)

    #Define loss function
    loss(x, y) = (size(trainingTargets,2) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y); 

    #Set Variables and Vectors
    lossTrainValues = Vector{Float32}()
    lossValValues = Vector{Float32}()
    lossTestValues = Vector{Float32}()
    numEpoch = 0
    numEpochsValidation = 0
    bestValAnn = ann
    savedTestLoss = NaN

    # Epoch 0 Losses
    (trainingLoss,validationLoss,testLoss) = calculateLossValues(loss,trainingDataset,
                                                                validationDataset=validationDataset,testDataset=testDataset)
    push!(lossTrainValues, trainingLoss) #Save epoch 0 loss train value
    push!(lossValValues, validationLoss) #Save epoch 0 loss validation value
    push!(lossTestValues, testLoss) #Save epoch 0 loss test value
    bestValidationLoss = validationLoss

    #Print Epoch 0 Data
    if showText
        println("---------------------------")
        println("Epoch: 0")
        println("Training loss value: ", trainingLoss)
        if !isempty(validationInputs)
            println("Validation loss value: ", validationLoss)
        end
    end

    while (numEpoch<maxEpochs) && (trainingLoss>minLoss) && (numEpochsValidation<maxEpochsVal)
        Flux.train!(loss, Flux.params(ann), [(trainingInputs', trainingTargets')], ADAM(learningRate)) # Network training
        numEpoch += 1
        
        #Loss Values
        (trainingLoss,validationLoss,testLoss) = calculateLossValues(loss,trainingDataset,
                                                                    validationDataset=validationDataset,testDataset=testDataset)
        push!(lossTrainValues, trainingLoss) #Save epoch loss train value
        push!(lossValValues, validationLoss) #Save epoch loss validation value
        push!(lossTestValues, testLoss) #Save epoch loss test value
        
        if !isempty(validationInputs)
            if (validationLoss<bestValidationLoss)
                bestValidationLoss = validationLoss;
                bestValAnn = deepcopy(ann);
                numEpochsValidation = 0;
                savedTestLoss = testLoss;
            else
                numEpochsValidation += 1;
            end
        end
        
        #Print Epoch n Data
        if showText
            println("---------------------------")
            println("Epoch: ", numEpoch)
            println("Training loss value: ", trainingLoss)
            if !isempty(validationInputs)
                println("Validation loss value: ", validationLoss)
                println("Best validation loss value: ", bestValidationLoss)
                println("Epochs without improvement: ", numEpochsValidation)
            end
        end
    end

    if !isempty(validationInputs)
        if showText #Print Final Model Data
            println("---------------------------")
            println("Training process completed!")
            println("Test loss value (Returned Model): ", savedTestLoss)
        end
        return (bestValAnn,(lossTrainValues,lossValValues,lossTestValues))
    else
        if showText #Print Final Model Data
            println("---------------------------")
            println("Training process completed!")
            println("Test loss value (Returned Model): ", testLoss)
        end
        return (ann,(lossTrainValues,lossTestValues))
    end
end;    

function trainClassANN(topology::AbstractArray{<:Int,1},  
                        trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}; 
                        validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}= 
                                    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)), 
                        testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}= 
                                    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)), 
                        transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
                        maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,  
                        maxEpochsVal::Int=20, showText::Bool=false)

    trainingTargets = reshape(trainingDataset[2], (:,1))
    validationTargets = reshape(validationDataset[2], (:,1))
    testTargets = reshape(testDataset[2], (:,1))

    return trainClassANN(topology,(trainingDataset[1],trainingTargets),validationDataset=(validationDataset[1],validationTargets),
                            testDataset=(testDataset[1],testTargets),transferFunctions=transferFunctions,maxEpochs=maxEpochs,
                            minLoss=minLoss,learningRate=learningRate,maxEpochsVal=maxEpochsVal,showText=showText)
end;

#### 4.1 ####

# Confusion Matrix:

function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    
    numInstances = length(targets);
    @assert (length(outputs)==numInstances) "Number of outputs and targets must match"

    TN = sum(.!outputs .& .!targets);
    FN = sum(.!outputs .&   targets);
    TP = sum(  outputs .&   targets);
    FP = sum(  outputs .& .!targets);

    confMatrix = [TN FP; FN TP]

    accuracy =  (TN + TP) / (TN + TP + FN + FP)
    errorRate = (FP + FN) / (TN + TP + FN + FP)
    sensitivity =     TP / (FN + TP)
    specificity =   TN / (FP + TN)
    PPV =           TP / (TP + FP)
    NPV =           TN / (TN + FN)

    if TN == numInstances
        sensitivity =     1.0
        PPV =           1.0
    elseif TP == numInstances
        specificity =   1.0
        NPV =           1.0
    end

    f_score = (2 * PPV * sensitivity) / (PPV + sensitivity)
       
    if sensitivity == 0 && PPV == 0
        f_score = 0.0
    end

    accuracy =      isnan(accuracy) ?       0.0 : accuracy
    errorRate =     isnan(errorRate) ?      0.0 : errorRate
    sensitivity =   isnan(sensitivity) ?      0.0 : sensitivity
    specificity =   isnan(specificity) ?    0.0 : specificity
    PPV =           isnan(PPV) ?            0.0 : PPV
    NPV =           isnan(NPV) ?            0.0 : NPV
    f_score =       isnan(f_score) ?        0.0 : f_score

    return (accuracy, errorRate, sensitivity, specificity, PPV, NPV, f_score, confMatrix)

end;

function confusionMatrix(outputs::AbstractArray{<:Real,1},targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    boolOutputs = outputs .>= threshold
    return confusionMatrix(boolOutputs, targets)
end;

function printConfusionMatrix(outputs::AbstractArray{Bool,1},targets::AbstractArray{Bool,1}; roundDigits::Int=4)
    
    @assert (roundDigits >= 0) "roundDigits must be greater than or equal to 0."

    (accuracy, errorRate, sensitivity, specificity, PPV, NPV, f_score, confMatrix) = confusionMatrix(outputs, targets)

    TN = confMatrix[1,1]
    FP = confMatrix[1,2]
    FN = confMatrix[2,1]
    TP = confMatrix[2,2]

    println("                   Pred")
    println("             ----------------")
    println("             |    -     +   |")
    println("     ------------------------")
    println("     |   -   |    $(TN)     $(FP)   |")
    println("Real ------------------------")
    println("     |   +   |    $(FN)     $(TP)   |")
    println("     ------------------------")
    println()
    println("Accuracy:                  $(round(accuracy,digits=roundDigits))")
    println("Error rate:                $(round(errorRate,digits=roundDigits))")
    println("Recall:                    $(round(sensitivity,digits=roundDigits))")
    println("Specificity:               $(round(specificity,digits=roundDigits))")
    println("Precision:                 $(round(PPV,digits=roundDigits))")
    println("Negative predictive value: $(round(NPV,digits=roundDigits))")
    println("F1-score:                  $(round(f_score,digits=roundDigits))")
end;

function printConfusionMatrix(outputs::AbstractArray{<:Real,1},targets::AbstractArray{Bool,1}; threshold::Real=0.5, roundDigits::Int=4)
    boolOutputs = outputs .>= threshold
    printConfusionMatrix(boolOutputs, targets, roundDigits=roundDigits)
end;

#### 4.2 ####

#One vs All
function fitANN(inputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2};
    topology::AbstractArray{<:Int,1}=Vector{Int}(), 
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)

    ann, = trainClassANN(topology, (inputs, targets), transferFunctions=transferFunctions, 
        maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate)

    return ann
end

function oneVSall(inputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; useSoftmax::Bool = false,
    topology::AbstractArray{<:Int,1}=Vector{Int}(), 
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)

    @assert size(inputs,1) == size(targets,1) "The number of patterns of inputs and targets must be the same" 
    
    numInstances = size(targets,1);
    numClasses = size(targets,2);
    
    outputs = Array{Float32,2}(undef, numInstances, numClasses);
    
    for numClass in 1:numClasses
        # Change fit() by the corresponding function in future assignments.
        model = fitANN(inputs, targets[:,[numClass]], topology=topology, transferFunctions=transferFunctions, 
            maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate);

        outputs[:,numClass] .= model(inputs')';
    end;
    
    if useSoftmax
        outputs = softmax(outputs')';
    end
    
    (_,indicesMaxEachInstance) = findmax(outputs, dims=2);
    outputs = falses(size(outputs));
    outputs[indicesMaxEachInstance] .= true;
    
    return outputs
end;

#Confusion Matrix

function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    
    numClasses = size(targets, 2)
    
    @assert (size(outputs,1) == size(targets,1)) "Number of patterns do not match in both arrays."
    @assert ((size(outputs,2) == numClasses) && (numClasses != 2)) "Number of columns do not match or there are 2 columns."

    if numClasses == 1
        return confusionMatrix(outputs[:,1], targets[:,1])
    end

    sensitivity =   zeros(numClasses)
    specificity =   zeros(numClasses)
    PPV =           zeros(numClasses)
    NPV =           zeros(numClasses)
    f_score =       zeros(numClasses)
    confMatrix =    zeros(Int64, size(targets,2), size(outputs,2))

    numInstancesFromEachClass = vec(sum(targets, dims=1));

    for numClass in findall(numInstancesFromEachClass .> 0)
        currentClassOutputs = outputs[:,numClass]
        currentClassTargets = targets[:,numClass]

        (_, _, currSensitivity, currSpecificity, currPPV, currNPV, currF_score, _) = 
            confusionMatrix(currentClassOutputs, currentClassTargets);
                
        sensitivity[numClass] = currSensitivity
        specificity[numClass] = currSpecificity
        PPV[numClass] =         currPPV
        NPV[numClass] =         currNPV
        f_score[numClass] =     currF_score
    end
    
    for numClassOutput in 1:numClasses
        for numClassTarget in 1:numClasses

            matches = outputs[:,numClassOutput] .&& targets[:,numClassTarget]
            matches = sum(matches)

            confMatrix[numClassTarget,numClassOutput] = matches
        end
    end

    if weighted

        weights = sum(targets, dims=1)

        sensitivity =  sum(sensitivity  .* weights') / sum(weights)
        specificity =  sum(specificity  .* weights') / sum(weights)
        PPV =          sum(PPV          .* weights') / sum(weights)
        NPV =          sum(NPV          .* weights') / sum(weights)
        f_score =      sum(f_score      .* weights') / sum(weights)

    else

        numClassesWithInstances = sum(numInstancesFromEachClass.>0);

        # Macro method.
        sensitivity =  sum(sensitivity) / numClassesWithInstances
        specificity =  sum(specificity) / numClassesWithInstances
        PPV =          sum(PPV)         / numClassesWithInstances
        NPV =          sum(NPV)         / numClassesWithInstances
        f_score =      sum(f_score)     / numClassesWithInstances
    end

    accuracyValue = accuracy(outputs,targets)
    errorRate = 1 - accuracyValue

    return (accuracyValue, errorRate, sensitivity, specificity, PPV, NPV, f_score, confMatrix)
end

function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    classifiedOutputs = classifyOutputs(outputs)
    return confusionMatrix(classifiedOutputs, targets, weighted=weighted)
end;

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    @assert (all(in.(outputs, [unique(targets)]))) "Not all the classes of 'outputs' are in 'targets'"
    
    classes = unique([targets..., outputs...])

    encodedOutputs = oneHotEncoding(outputs, classes)
    encodedTargets = oneHotEncoding(targets, classes)

    return confusionMatrix(encodedOutputs, encodedTargets, weighted=weighted)
end;

### 5 ###

### Cross-validation

function crossvalidation(N::Int64, k::Int64)
    kfolds = collect(1:k)
    numberOfRepeats = ceil(Int64, N/k)
    transformed = repeat(kfolds, numberOfRepeats)    
    return Random.shuffle!(transformed[1:N])
end

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    patternsIndexes = Vector{Int64}(undef, size(targets,1))
    Ni = sum(targets,dims=1) # Vector with the number of patterns for each class.
    @assert all(Ni .>= k) "All classes must hace at least k patterns."

    for numClass = 1:size(targets,2)
        patternsIndexes[targets[:,numClass]] .= crossvalidation(Ni[numClass], k)
    end

    return patternsIndexes
end

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    classes = unique(targets)
    encodedTargets = targets .== permutedims(classes)
    return crossvalidation(encodedTargets, k)
end

### Train classes

function trainClassANN(topology::AbstractArray{<:Int,1}, 
        trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, 
        kFoldIndices::	Array{Int64,1}; 
        transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
        maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, repetitionsTraining::Int=1, 
        validationRatio::Real=0.0, maxEpochsVal::Int=20, showText::Bool=false)

    @assert maxEpochs >= 1 "The maximum number of epochs has to be at least 1"
    @assert maxEpochsVal >= 1 "The maximum number of validation epochs has to be at least 1"
    @assert ((validationRatio>=0.) & (validationRatio<=1.)) "validationRatio must take a value between 0 and 1"

    (trainingInputs, trainingTargets) = trainingDataset;

    @assert (size(trainingInputs, 1)==size(trainingTargets, 1)) "The number of patterns in the training sets do not match"

    numFolds = maximum(kFoldIndices);

    testAccuracies =    Array{Float64,1}()
    testSens =          Array{Float64,1}()
    testSpec =          Array{Float64,1}()
    testPPV =           Array{Float64,1}()
    testNPV =           Array{Float64,1}()
    testF1 =            Array{Float64,1}()

    for numFold in 1:numFolds
        trainingInputsK =    trainingInputs[kFoldIndices.!=numFold,:];
        testInputsK =        trainingInputs[kFoldIndices.==numFold,:];
        trainingTargetsK =  trainingTargets[kFoldIndices.!=numFold,:];
        testTargetsK =      trainingTargets[kFoldIndices.==numFold,:];
        
        testAccuraciesEachRepetition =  Array{Float64,1}();
        testSensEachRepetition =        Array{Float64,1}();
        testSpecEachRepetition =        Array{Float64,1}();
        testPPVEachRepetition =         Array{Float64,1}();
        testNPVEachRepetition =         Array{Float64,1}();
        testF1EachRepetition =          Array{Float64,1}();
        
        for numTraining in 1:repetitionsTraining
            
            if validationRatio > 0
                # Train ANN using training, validation and test sets.
                (trainingIndices, validationIndices) = holdOut(size(trainingInputsK,1), validationRatio);

                trainInputs =   trainingInputsK[trainingIndices,:]
                trainTargets = trainingTargetsK[trainingIndices,:]
                valInputs =     trainingInputsK[validationIndices,:]
                valTargets =   trainingTargetsK[validationIndices,:]
                
                ann, = trainClassANN(topology,(trainInputs,trainTargets),validationDataset=(valInputs,valTargets),
                                    testDataset=(testInputsK,testTargetsK),transferFunctions=transferFunctions, 
                                    maxEpochs=maxEpochs, minLoss=minLoss,learningRate=learningRate,
                                    maxEpochsVal=maxEpochsVal,showText=showText)
            else
                # Train ANN using training and test sets.
                ann, = trainClassANN(topology,(trainingInputsK,trainingTargetsK),testDataset=(testInputsK,testTargetsK),
                                    transferFunctions=transferFunctions, maxEpochs=maxEpochs, minLoss=minLoss, 
                                    learningRate=learningRate,showText=showText)
            end;

            # Calculate metrics with confussionMatrix() function
            (acc,_,sensitivity,specificity,PPV,NPV,f1,_) = confusionMatrix(ann(testInputsK')',testTargetsK)

            push!(testAccuraciesEachRepetition, acc)
            push!(testSensEachRepetition,       sensitivity)
            push!(testSpecEachRepetition,       specificity)
            push!(testPPVEachRepetition,        PPV)
            push!(testNPVEachRepetition,        NPV)
            push!(testF1EachRepetition,         f1)

        end;

        #K-Folds Metrics
        push!(testAccuracies,   mean(testAccuraciesEachRepetition))
        push!(testSens,         mean(testSensEachRepetition))
        push!(testSpec,         mean(testSpecEachRepetition))
        push!(testPPV,          mean(testPPVEachRepetition))
        push!(testNPV,          mean(testNPVEachRepetition))
        push!(testF1,           mean(testF1EachRepetition))
        
    end; # kfold loop end

    finalAccuracy = (mean(testAccuracies),  std(testAccuracies))
    finalSens =     (mean(testSens),        std(testSens))
    finalSpec =     (mean(testSpec),        std(testSpec))
    finalPPV =      (mean(testPPV),         std(testPPV))
    finalNPV =      (mean(testNPV),         std(testNPV))
    finalF1 =       (mean(testF1),          std(testF1))

    if showText
        println("---------------------------")
        for k in 1:numFolds
        println("K",k," Accuracy: ", testAccuracies[k])
        end
    end

    return (finalAccuracy,finalSens,finalSpec,finalPPV,finalNPV,finalF1)
end;

function trainClassANN(topology::AbstractArray{<:Int,1},
        trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}},
        kFoldIndices::	Array{Int64,1};
        transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
        maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,repetitionsTraining::Int=1, 
        validationRatio::Real=0.0, maxEpochsVal::Int=20, showText::Bool=false)

    matrixTargets = reshape(trainingDataset[2], (:,1))

    return trainClassANN(topology,(trainingDataset[1],matrixTargets),kFoldIndices,
        transferFunctions=transferFunctions,maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate,
        repetitionsTraining=repetitionsTraining, validationRatio=validationRatio, maxEpochsVal=maxEpochsVal, showText=showText)
end;

### 6 ###

### Crossvalidation

function modelCrossValidation(modelType::Symbol,
        modelHyperparameters::Dict,
        inputs::AbstractArray{<:Real,2},
        targets::AbstractArray{<:Any,1},
        crossValidationIndices::Array{Int64,1})
    
    @assert modelType in (:ANN, :SVM, :DecisionTree, :kNN) "The selected modelType is not available"
    @assert size(inputs,1) == size(targets,1) "The number of patterns has to be equal in the input and target sets"
    @assert size(inputs,1) == length(crossValidationIndices) "Number of indexes has to match with the number of patterns"

    if modelType == :ANN

        println("Executing for ANN.")
    
        encodedTargets = oneHotEncoding(targets, unique(targets))

        return trainClassANN(modelHyperparameters["topology"], (inputs,encodedTargets), crossValidationIndices, 
            transferFunctions   = modelHyperparameters["transferFunctions"],
            maxEpochs           = modelHyperparameters["maxEpochs"],
            minLoss             = modelHyperparameters["minLoss"],
            learningRate        = modelHyperparameters["learningRate"],
            repetitionsTraining = modelHyperparameters["repetitionsTraining"],
            validationRatio     = modelHyperparameters["validationRatio"],
            maxEpochsVal        = modelHyperparameters["maxEpochsVal"],
            showText            = modelHyperparameters["showText"]
        );
        
    elseif modelType == :SVM

        println("Executing for SVM.")

        model = SVC(
            kernel          = modelHyperparameters["kernel"], 
            degree          = modelHyperparameters["degree"], 
            gamma           = modelHyperparameters["gamma"], 
            C               = modelHyperparameters["C"]
        );

    elseif modelType == :DecisionTree

        println("Executing for DecisionTree.")

        model = DecisionTreeClassifier(
            max_depth       = modelHyperparameters["max_depth"], 
            random_state    = modelHyperparameters["random_state"]
        );

    elseif modelType == :kNN
        
        println("Executing for kNN.")

        model = KNeighborsClassifier(modelHyperparameters["k"]);
    end

    numFolds = maximum(crossValidationIndices);

    testAccuracies =    Array{Float64,1}()
    testSens =          Array{Float64,1}()
    testSpec =          Array{Float64,1}()
    testPPV =           Array{Float64,1}()
    testNPV =           Array{Float64,1}()
    testF1 =            Array{Float64,1}()

    for numFold in 1:numFolds
        trainingInputsK =    inputs[crossValidationIndices.!=numFold,:];
        testInputsK =        inputs[crossValidationIndices.==numFold,:];
        trainingTargetsK =  targets[crossValidationIndices.!=numFold,:];
        testTargetsK =      targets[crossValidationIndices.==numFold,:];

        fit!(model, trainingInputsK, trainingTargetsK)
        
        # Calculate metrics with confussionMatrix() function
        (acc,_,sensitivity,specificity,PPV,NPV,f1,_) = confusionMatrix(predict(model, testInputsK),testTargetsK[:,1])

        push!(testAccuracies,   acc)
        push!(testSens,         sensitivity)
        push!(testSpec,         specificity)
        push!(testPPV,          PPV)
        push!(testNPV,          NPV)
        push!(testF1,           f1)
    
    end
    
    finalAccuracy = (mean(testAccuracies),  std(testAccuracies))
    finalSens =     (mean(testSens),        std(testSens))
    finalSpec =     (mean(testSpec),        std(testSpec))
    finalPPV =      (mean(testPPV),         std(testPPV))
    finalNPV =      (mean(testNPV),         std(testNPV))
    finalF1 =       (mean(testF1),          std(testF1))
    
    return (finalAccuracy,finalSens,finalSpec,finalPPV,finalNPV,finalF1)

end

### 7 ###

### Ensemble models

function trainClassEnsemble(estimators::AbstractArray{Symbol,1}, 
        modelsHyperParameters::AbstractArray{Dict{String,Any}, 1},     
        trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},    
        kFoldIndices::Array{Int64,1},
        weights::Array{Int64,1}
    )

    # Ensemble chosen: Weigthed Mayority Voting

    (inputs, targets) = trainingDataset

    @assert length(estimators) == length(modelsHyperParameters) "There must be hyperparameters for all estimators"
    @assert length(estimators) == length(weights) "Weights must be specified for all estimators"
    @assert length(kFoldIndices) == size(inputs,1) "The number of KFoldIndices must be equal to the number of patterns"
    @assert size(targets,1) == size(inputs,1) "The number of patterns in the inputs and targets has to be equal"

    numFolds = length(unique(kFoldIndices))
    numModels   = length(estimators)
    baseModels = []

    testAccuracies      = zeros(Float64, numFolds)
    testSens            = zeros(Float64, numFolds)
    testSpec            = zeros(Float64, numFolds)
    testPPV             = zeros(Float64, numFolds)
    testNPV             = zeros(Float64, numFolds)
    testF1              = zeros(Float64, numFolds)

    for numFold in 1:numFolds
        
        # Prepare the data in train-test split with CV indices (K-fold datasets)
        trainingInputsK     = inputs[kFoldIndices.!=numFold,:];
        testInputsK         = inputs[kFoldIndices.==numFold,:];
        trainingTargetsK    = targets[kFoldIndices.!=numFold,:];
        testTargetsK        = targets[kFoldIndices.==numFold,:];
        
        for (index,estimator) in enumerate(estimators)
            
            modelHyperparameters = modelsHyperParameters[index]
            
            # Build base model
            if estimator == :SVM

                model = SVC(
                    kernel  = modelHyperparameters["kernel"], 
                    degree  = modelHyperparameters["kernelDegree"], 
                    gamma   = modelHyperparameters["kernelGamma"], 
                    C       = modelHyperparameters["C"],
                    probability = true
                )

            elseif estimator == :DecisionTree

                model = DecisionTreeClassifier(
                    max_depth       = modelHyperparameters["maxDepth"], 
                    random_state    = modelHyperparameters["randomState"]
                )

            elseif estimator == :kNN

                model = KNeighborsClassifier(modelHyperparameters["k"])

            elseif estimator == :MLPC # == ANN

                model = MLPClassifier(
                    hidden_layer_sizes  = modelHyperparameters["topology"],
                    max_iter            = modelHyperparameters["maxEpochs"],
                    learning_rate_init  = modelHyperparameters["learningRate"]
                )

            end
            
            #Train Model 
            println(size(trainingInputsK))
            println(size(trainingTargetsK))
            fit!(model, trainingInputsK, trainingTargetsK);
            
            # Store model
            push!(baseModels, (string(estimator), model) )

        end
        
        # Choose a classifier method: Weigthed Mayority Voting.
        println(baseModels)
        ensemble_model = VotingClassifier(estimators = baseModels, voting="soft", weights=weights) # n_jobs = -1
        fit!(ensemble_model, trainingInputsK, trainingTargetsK)
        
        # Predict using the test set.
        testOutputsK = predict(ensemble_model, testInputsK)

        # Obtaining metrics
        (acc,_,sensitivity,specificity,PPV,NPV,f1,_) = confusionMatrix(testOutputsK,testTargetsK[:,1])

        push!(testAccuracies,   acc)
        push!(testSens,         sensitivity)
        push!(testSpec,         specificity)
        push!(testPPV,          PPV)
        push!(testNPV,          NPV)
        push!(testF1,           f1)

        accuracy = score(ensemble_model, testInputsK, testTargetsK)
        # testResults[numFold] = accuracy
        
        println("K", numFold, " Accuracy: ", round(accuracy, digits = 4))
    end
        
    # Finally, provide the result of averaging the values of these vectors for each metric together with their standard deviations.

    finalAccuracyMean = mean(testAccuracies)
    finalSensMean =     mean(testSens)
    finalSpecMean =     mean(testSpec)
    finalPPVMean =      mean(testPPV)
    finalNPVMean =      mean(testNPV)
    finalF1Mean =       mean(testF1)

    finalAccuracyStd = std(testAccuracies)
    finalSensStd =     std(testSens)
    finalSpecStd =     std(testSpec)
    finalPPVStd =      std(testPPV)
    finalNPVStd =      std(testNPV)
    finalF1Std =       std(testF1)

    return ([finalAccuracyMean, finalSensMean, finalSpecMean, finalPPVMean, finalNPVMean, finalF1Mean],
        [finalAccuracyStd, finalSensStd, finalSpecStd, finalPPVStd, finalNPVStd, finalF1Std])

    # finalAcc = (mean(testResults), std(testResults))

    # return finalAcc
end

function trainClassEnsemble(baseEstimator::Symbol, 
        modelsHyperParameters::Dict,
        trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},     
        kFoldIndices::Array{Int64,1};
        NumEstimators::Int=100
    )

    @assert length(NumEstimators) > 0 "Number of estimators must be at least one"

    repeated_estimator = repeat([baseEstimator], NumEstimators)
    repeated_HyperParameters = repeat([modelsHyperParameters], NumEstimators)
    weights = ones(Int64,NumEstimators)

    return trainClassEnsemble(repeated_estimator, repeated_HyperParameters, trainingDataset, kFoldIndices, weights)

    #TODO: Check if it's working properly
end

### 8 ###

### Dimension reduction



# LAST UPDATE: Notebook 7
