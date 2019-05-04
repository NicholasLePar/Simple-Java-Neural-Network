import java.util.*;
import java.io.*;

public class NeuralNetwork implements java.io.Serializable {
	//number of hidden layers in neural network
	int hidden_layers;
	//number of total layers in neural network (should be hidden_layers+1, 
	//since accounting for output layer)
	int total_layers;
	//An array where the index represents the layer, and the value represents the number of neurons
	// at that layer
	int[] neurons_per_layer;
	//The 2d array of neuron objects that represents the neural network
	//where the index [i][] represents layer i of the neural network, 
	//and index[i][j] represents the neuron j at layer i.
	//an individual neuron object is found at index [i][j]
	Neuron[][] neural_network;





	//Consturctor for a Neural network which sets: the number of total and hidden layers, 
	//neurons per layer(including output layer), 
	//and the type of each neuron per layer (including output).
	public NeuralNetwork(int[][] neuron_type_per_layer, 
							double[][][] weight_per_neuron_edge,
							double[][] bias_per_neuron,
							double[] input_layer_values){

		//the number of hidden layers (subtract 1 for final output layer)
		this.hidden_layers=neuron_type_per_layer.length-1;
		//the number of total layers, (includes the output layer)
		this.total_layers = neuron_type_per_layer.length;

		//the number of neurons in each hidden layer;
		this.neurons_per_layer = new int[total_layers];
		for(int i=0; i<total_layers;i++){
			this.neurons_per_layer[i] = neuron_type_per_layer[i].length;
		}

		//generate Neural Network structure
		this.neural_network = new Neuron[total_layers][];
		for(int i=0; i<total_layers; i++){
			this.neural_network[i] = new Neuron[neurons_per_layer[i]];
			for(int j=0; j<neurons_per_layer[i]; j++){
				//ReLU generation
				if(neuron_type_per_layer[i][j]==0){
					//if bottom hidden_layer inputs shall come from input layer
					if(i==0){
						neural_network[i][j] = new ReLU(input_layer_values,
														weight_per_neuron_edge[i][j],
														bias_per_neuron[i][j]);
					}
					//otherwise input to neuron should come from every neuron on layer below
					else{
						double[] layer_output_to_input = new double[neural_network[i-1].length];
						for(int k=0; k<layer_output_to_input.length; k++){
							layer_output_to_input[k] = neural_network[i-1][k].getOutput();
						}
						neural_network[i][j] = new ReLU(layer_output_to_input,
														weight_per_neuron_edge[i][j],
														bias_per_neuron[i][j]);
					}
				}
				//Sigmoid generation
				else if(neuron_type_per_layer[i][j]==1){
					//if bottom hidden_layer inputs shall come from input layer
					if(i==0){
						neural_network[i][j] = new Sigmoid(input_layer_values,
															weight_per_neuron_edge[i][j],
															bias_per_neuron[i][j]);
					}
					//otherwise input to neuron should come from every neuron on layer below
					else{
						double[] layer_output_to_input = new double[neural_network[i-1].length];
						for(int k=0; k<layer_output_to_input.length; k++){
							layer_output_to_input[k] = neural_network[i-1][k].getOutput();
						}
						neural_network[i][j] = new Sigmoid(layer_output_to_input,
															weight_per_neuron_edge[i][j],
															bias_per_neuron[i][j]);
					}
				}
			}
		}
	}





	//Set the input layer values and propagate the change up stream through the Network
	public void setInputs(double[] inputs){
		for(int i=0; i<neurons_per_layer.length; i++){
			for(int j=0; j<neurons_per_layer[i]; j++){
				//if the first layer then set the new inputs directly
				if(i==0){
					neural_network[i][j].setInputs(inputs);
				}
				//otherwise propagate the input change from the ouput of lower layers as
				//the input of higher layers
				else{
					double[] layer_output_to_input = new double[neural_network[i-1].length];
					for(int k=0; k<layer_output_to_input.length; k++){
						layer_output_to_input[k] = neural_network[i-1][k].getOutput();
					}
					neural_network[i][j].setInputs(layer_output_to_input);
				}
			}
		}
	}





	//Return number of total layers (all hidden + output layer)
	public int getTotalLayers(){
		return total_layers;
	}





	//Return number of hidden layers in neural network.
	public int getHiddenLayers(){
		return hidden_layers;
	}





	//Return an array that has the neurons per layer
	public int[] getNeuronsPerLayer(){
		return neurons_per_layer;
	}





	//Return the output of the entire neural network
	public double getOutput(){
		return neural_network[total_layers-1][0].getOutput();
	}





	//Return the weights of all edges going into each neuron
	public double[][][] getWeights(){
		double [][][] weightsOfNeuronsPerLayer = new double [total_layers][][];
		for(int i=0; i<total_layers; i++){
			weightsOfNeuronsPerLayer[i] = new double [neurons_per_layer[i]][];
			for(int j=0; j<neurons_per_layer[i]; j++){
				weightsOfNeuronsPerLayer[i][j] = neural_network[i][j].getWeights();
			}
		}
		return weightsOfNeuronsPerLayer;
	}





	//Backpropagate one step of Schocastic Gradient desecent to training Neural Network
	public void oneStepSGD(double nStepSize, double expectedOutput){
		//output of each neurons SGD. Used for referencing in backpropagation.
		double[][][][] sgdOutput = new double[total_layers][][][];

		//Preform backward propagation of SGD on Neural Network. (output layer first)
		for(int i=total_layers-1; i>-1; i--){

			//the number of neurons per layer
			sgdOutput[i] = new double[neurons_per_layer[i]][][];
			//perform SGD on output layer
			//output layer has no additional output weight or derivative with respect to error
			//from a higher layer, thus pass in 1 for dEuC and outputWeight
			if(i==total_layers-1){
				sgdOutput[i][0] = neural_network[i][0].oneStepSGD(nStepSize,1,1,true,expectedOutput);
			}

			//Intermideate layer SGD
			else{
				//SGD for every parent neuron
				for(int j=0;j<neurons_per_layer[i+1];j++){
					//on every neuron in the layer
					for(int k=0; k<neurons_per_layer[i]; k++){
						//perform SGD with Output weight and partial derivate with respect
						// to error of the intermideate layer coming from previous SGD.
						sgdOutput[i][k] = neural_network[i][k].oneStepSGD(nStepSize,
																			sgdOutput[i+1][j][0][0],
																			sgdOutput[i+1][j][1][k],false,0);
					}
				}				
			}
		}
	}





	//Run Neural Network training until error reaches minimum and begins to rise
	//on the eval data set.
	public void trainToMinimumError(ArrayList<Double> trainingData, ArrayList<Double> evalData, double stepSize){
		//epoch counter
		int epoch = 0;
		//the error from the previous epoch
		double previousError = Double.MAX_VALUE;
		//the error of the current epoch for the comparison
		double errorAfterEpoch = Double.MAX_VALUE;
		while(errorAfterEpoch<=previousError){
			//Epoch counter
			epoch++;
			//set the previous error for terminal training check.
			previousError = errorAfterEpoch;
			errorAfterEpoch=0;
			//train over the entire trainging set (one epoch)
			for(int index=0; index<trainingData.size(); index=index+3){
				double[] newInputs = new double[2];
				//x1 students hw score(from training data set)
				newInputs[0] = trainingData.get(index);
				//x2 students midterm score(from training data set)
				newInputs[1] = trainingData.get(index+1);
				//y being whethecr they recieved an A or not(from training data set)
				double y = trainingData.get(index+2);
				//Set the neural network with new inputs from training data
				this.setInputs(newInputs);
				//Preform one step of SGD
				this.oneStepSGD(stepSize,y);
			}
			//Evaluate total error of current weights from the training set
			//against the entire evaluation set
			for(int index=0; index<evalData.size(); index=index+3){
				double[] newInputs = new double[2];
				//x1 students hw score(from eval data set)
				newInputs[0] = evalData.get(index);
				//x2 students midterm score(from eval data set)
				newInputs[1] = evalData.get(index+1);
				//y being whethecr they recieved an A or not(from eval data set)
				double y = evalData.get(index+2);
				//Set the neural network with new inputs from eval data
				this.setInputs(newInputs);
				//calculate the ouput of the neural network to determine the error
				double outputFromEval = this.getOutput();
				//Error of the neural network needs to be summed as a total across
				//all eval data
				errorAfterEpoch += 0.5*Math.pow((outputFromEval-y),2);
			}

			//OUTPUT PER EPOCH
			//weights of each neuron after one epoch of training
			System.out.println("WEIGHTS AFTER EPOCH:" + epoch);
			double [][][] test_neuron_weights = this.getWeights();
			for(int i=0; i<test_neuron_weights.length; i++){
				System.out.println("Layer " + i + ":");
				for(int j=0; j<test_neuron_weights[i].length; j++){
					System.out.print("Neuron " + j + " has the weights:");
					for(int k=0; k<test_neuron_weights[i][j].length;k++){
						System.out.print(String.format("%.5f",test_neuron_weights[i][j][k]) +",");
					}
					System.out.print("end\n");
				}
			}
			//print the error of the epoch
			System.out.println("Error from eval set: "+
								String.format("%.5f",errorAfterEpoch));
		}
	}
	




	//Run Neural Network against a test set and display its accuracy
	public void runTestSet(ArrayList<Double> testData,double accuracyThreshold){
		//compare the output of the Neural network against the test data labels
		//determine accuracy of neural network
		double predictions = 0;
		double correctPredictions = 0;
		for(int index=0; index<testData.size(); index=index+3){
			double[] newInputs = new double[2];
			//x1 students hw score(from test data set)
			newInputs[0] = testData.get(index);
			//x2 students midterm score(from test data set)
			newInputs[1] = testData.get(index+1);
			//y being whether they recieved an A or not(from test data set)
			double y = testData.get(index+2);
			//Set the neural network with new inputs from training data
			this.setInputs(newInputs);
			//Get the output of the neural network from test inputs
			double outputFromTest = this.getOutput();
			//determine if the output is a correct prediction based on the threshold
			if(outputFromTest>=accuracyThreshold){
				predictions++;
				if(y==1){
					correctPredictions++;
				}
			}
			else{
				predictions++;
				if(y==0){
					correctPredictions++;
				}
			}
		}
		//calculate the accuracy of the neural network.
		double accuracy = correctPredictions/predictions;
		//Print the accuracy

		System.out.println( "Accuracy of Neural Network with threshold " + 
								accuracyThreshold +
								" is: "+
								String.format("%.5f",accuracy));
	}




	//Debug main method.
	public static void main (String[] args){
		System.out.println("NeuralNetwork Main");
		
		//Create neural network from neural network setup file
		if(args.length!=1){
			System.out.println("ERROR: Must pass in network setup file.");
			return;
		} 

		//Neural Network Parameters
		double [] inputLayer;
		int [][] neuronTypePerLayer;
		double [][][] weightPerNeuronEdge;
		double [][] biasPerNeuron;			

		///NETWORK CONFIG INTERPRETOR############################
    	ArrayList<Double> networkConfig = new ArrayList<Double>();
    	try{
    		String networkSetupFile=args[0];
    	    File f = new File(networkSetupFile);
    	    Scanner sc = new Scanner(f);
    	    //CONFIG VARIABLES 
			int numberOfInputs=0;
			int numberOfLayers=0;
			inputLayer = new double[0];
			neuronTypePerLayer = new int[0][0];
			weightPerNeuronEdge = new double[0][0][0];
			biasPerNeuron = new double[0][0];
    	    //COUNT VARIABLES
    	    int lineCount=0;
    	    int layerCount=0;
    	    int neuronCount=0;
    	    while(sc.hasNext()){
    	        
    	        //read in line of the config file
    	        String configLine = sc.nextLine();
    	        lineCount++;
    	        //LINE CLEANING
    	        configLine = configLine.replaceAll("\\s+","");
    	    	if(configLine.equals("")){
    	    		continue;
    	    	}

    	        //COMMENT DISCARD LINE
    	        if(configLine.charAt(0) == '/'){
    	        	if(configLine.charAt(1)=='/'){
    	        		continue;
    	        	}
    	        	System.out.println("ERROR("+lineCount+"): bad config cmd");
    	        	return;
    	        }

    	        //COMMAND CLEAN
    	        String[] cmdLine = configLine.split(":");
				String cmd = cmdLine[0]; 
				String [] cmdVariables = cmdLine[1].split(",");
				
				//VALID ENTRIES
    	        if(cmd.equals("INPUT")){
					try{
    	        		System.out.println("***INPUT***");
						numberOfInputs = cmdVariables.length;
						if(numberOfInputs<=0){
							System.out.println("ERROR("+lineCount+"): bad config variables");
							return;
						}
						System.out.println("Number of Inputs:" + numberOfInputs);
						inputLayer = new double[numberOfInputs];
						System.out.print("Value of Inputs:");
						for(int i=0; i<numberOfInputs; i++){
							inputLayer[i] = Double.parseDouble(cmdVariables[i]);
							System.out.print(inputLayer[i]+",");	
						}
						System.out.print("end\n");	
					}catch(NumberFormatException ex){
						System.out.println("ERROR("+lineCount+"): NOI not set correctly to an integer");
						System.out.println(ex.getMessage());
						return;
					}
    	        }
    	        else if(cmd.equals("NPL")){
    	        	try{
    	        		System.out.println("***NPL***");
    	        		numberOfLayers = cmdVariables.length;
    	        		if(numberOfLayers<=0){
							System.out.println("ERROR("+lineCount+"): bad config variables");
							return;
						}
						System.out.println("Number of Layers:" + numberOfLayers);
						//set config variables arrays
						neuronTypePerLayer = new int[numberOfLayers][];
						biasPerNeuron = new double[numberOfLayers][];
						weightPerNeuronEdge = new double[numberOfLayers][][];
						System.out.println("Number of Neurons per Layer");
						for(int i=0; i<numberOfLayers; i++){
							int neuronsPerLayer = Integer.parseInt(cmdVariables[i]);
							neuronTypePerLayer[i] = new int[neuronsPerLayer];
							biasPerNeuron[i] = new double[neuronsPerLayer];
							weightPerNeuronEdge[i] = new double[neuronsPerLayer][];
							System.out.println("Layer " + (i+1) + ": " + neuronsPerLayer + " neurons");	
						}
    	        	}catch(NumberFormatException ex){
    	        		System.out.println("ERROR("+lineCount+"): NOI not set correctly to an integer");
						System.out.println(ex.getMessage());
						return;
    	        	}
    	        }
    	        else if(cmd.equals("LAYER")){
					try{
						System.out.println("***LAYER***");
						//if the count of LAYER cmds is greater than number of layers return error
						//it is layerCount+1 because layerCount starts at 0 and used for indexing
						if((layerCount+1)>numberOfLayers){
							System.out.println("ERROR("+lineCount+"): LAYER cmd exceeds total number of layers stated in NPL");
							return;
						}
						//Check if the number of neurons in the layer matches what was stated in the NPL cmd
						int numberOfNeuronsInLayer = cmdVariables.length;
						if(numberOfNeuronsInLayer!=neuronTypePerLayer[layerCount].length){
							System.out.println("ERROR("+lineCount+"): bad config variables, layer neurons do not match NPL");
							return;
						}
						System.out.println("LAYER " + (layerCount+1));
						System.out.println("Number of neurons in layer:" + numberOfNeuronsInLayer);
						System.out.print("Neuron type in layer:");
						for(int i=0; i<numberOfNeuronsInLayer; i++){
							int neuron = Integer.parseInt(cmdVariables[i]);
							neuronTypePerLayer[layerCount][i] = neuron;
							System.out.print(neuron+",");	
						}
						System.out.print("end\n");	
						//Increment layer count
						layerCount++;
					}catch(NumberFormatException ex){
						System.out.println("ERROR("+lineCount+"): NOI not set correctly to an integer");
						System.out.println(ex.getMessage());
						return;
					}
    	        }
    	        else if(cmd.equals("BIAS")){
    	        	try{
	    	        	System.out.println("***BIAS***");
	    	        	if(cmdVariables.length!=1){
							System.out.println("ERROR("+lineCount+"): bad bias variable");
							return;
	    	        	}
	    	        	double bias = Double.parseDouble(cmdVariables[0]);
	    	        	//Layer count need to be -1 because it is incremented in LAYER call
	    	        	biasPerNeuron[layerCount-1][neuronCount] = bias;
	    	        	System.out.println("Bias for Layer " + layerCount + " neuron " + (neuronCount+1) + ": " + bias);
    	        	}catch(NumberFormatException ex){
						System.out.println("ERROR("+lineCount+"): BIAS not set correctly to a double");
						System.out.println(ex.getMessage());
						return;
					}
    	        }
    	        else if(cmd.equals("WEIGHT")){
    	        	try{
    	        		System.out.println("***WEIGHT***");
    	        		if(layerCount==1){
    	        			//in first layer fail if weights are less than input layer or more than input layer plus
    	        			// a bias (numberOfInputs + 1)
    	        			if((numberOfInputs+1)<cmdVariables.length || cmdVariables.length<numberOfInputs){
    	        				System.out.println("ERROR("+lineCount+"): bad weight variables");
								return;
    	        			}
    	        			//Set the edge weights for the neuron
    	        			System.out.print("Edge weights for Layer " + layerCount + " neuron " + neuronCount 
    	        								+ ": ");
    	        			weightPerNeuronEdge[layerCount-1][neuronCount] = new double[cmdVariables.length];
    	        			for(int i=0;i<cmdVariables.length;i++){
    	        				double edgeWeight = Double.parseDouble(cmdVariables[i]);
    	        				weightPerNeuronEdge[layerCount-1][neuronCount][i] = edgeWeight;
    	        				System.out.print(edgeWeight+",");
    	        			}
    	        			System.out.print("end\n");
    	        		}else{
    	        			//Layer count is the indexer for LAYER. So it's value outside of LAYER is the actual layer
    	        			//so to index the actual layer with layerCount it would be:(layerCount-1)
    	        			//to access the layer below the current layer it would be: (layerCount-2)
    	        			int neuronsInLayerBelow = weightPerNeuronEdge[layerCount-2].length;
    	        			//for all layers fail if weights are less than the number of neurons below or more than
    	        			// number of neurons below plus a bias(neuronsInLayerBelow + 1)
    	        			if((neuronsInLayerBelow+1)<cmdVariables.length || cmdVariables.length<neuronsInLayerBelow){
    	        				System.out.println("ERROR("+lineCount+"): bad weight variables");
								return;
    	        			}
    	        			//Set the edge weights for the neuron
    	        			System.out.print("Edge weights for Layer " + layerCount + " neuron " + neuronCount 
    	        								+ ": ");
    	        			weightPerNeuronEdge[layerCount-1][neuronCount] = new double[cmdVariables.length];
    	        			for(int i=0;i<cmdVariables.length;i++){
    	        				double edgeWeight = Double.parseDouble(cmdVariables[i]);
    	        				weightPerNeuronEdge[layerCount-1][neuronCount][i] = edgeWeight;
    	        				System.out.print(edgeWeight+",");
    	        			}
    	        			System.out.print("end\n");
    	        		}
    	        		//increment neuron count based on the layer you are in, this allows for zeroing neuronCount
	    	        	//when starting the next layer
	    	        	neuronCount = (neuronCount+1) % biasPerNeuron[layerCount-1].length;
    	        	}catch(NumberFormatException ex){
						System.out.println("ERROR("+lineCount+"): WEIGHT not set correctly to a double");
						System.out.println(ex.getMessage());
						return;
					}
    	        }

    	    }
    	}
    	catch(FileNotFoundException ex){
    		System.out.println("ERROR: Network setup file not found.");
    		return;
    	}
    	//NETWORK CONFIG INTERPRETOR END#########################

    	
		//CREATE TRAINING DATA SETS#######################################
		//create an arraylist from the final test set
		String filenameTest = "./hw2_midterm_A_test.txt";
    	ArrayList<Double> testData = new ArrayList<Double>();
    	try{
    	    File f = new File(filenameTest);
    	    Scanner sc = new Scanner(f);
    	    while(sc.hasNext()){
    	        if(sc.hasNextDouble()){
    	            double i = sc.nextDouble();
    	            testData.add(i);
    	        }
    	        else{
    	            sc.next();
    	        }
    	    }
    	}
    	catch(FileNotFoundException ex){
    		System.out.println("File Not Found.");
    	}
    	//Create arraylist from the evaluation set.
		String filenameEval = "./hw2_midterm_A_eval.txt";
    	ArrayList<Double> evalData = new ArrayList<Double>();
    	try{
    	    File f = new File(filenameEval);
    	    Scanner sc = new Scanner(f);
    	    while(sc.hasNext()){
    	        if(sc.hasNextDouble()){
    	            double i = sc.nextDouble();
    	            evalData.add(i);
    	        }
    	        else{
    	            sc.next();
    	        }
    	    }
    	}
    	catch(FileNotFoundException ex){
    	    System.out.println("File Not Found.");
    	}
    	//Create arraylist from the training data set.
		String filenameTraining = "./hw2_midterm_A_train.txt";
    	ArrayList<Double> trainingData = new ArrayList<Double>();
    	try{
    	    File f = new File(filenameTraining);
    	    Scanner sc = new Scanner(f);
    	    while(sc.hasNext()){
    	        if(sc.hasNextDouble()){
    	            double i = sc.nextDouble();
    	            trainingData.add(i);
    	        }
    	        else{
    	            sc.next();
    	        }
    	    }
    	}
    	catch(FileNotFoundException ex){
    	    System.out.println("File Not Found.");
    	}
		//END OF DATA SET CREATION#######################################



		//consturct the neural network
		NeuralNetwork a_NN = new NeuralNetwork(neuronTypePerLayer,
												weightPerNeuronEdge,
												biasPerNeuron,
												inputLayer);

		//Step size for stepping in SGD
		double testStepSize = 0.1;
		
		//Train neural network with training data set.
		//run training until error against the eval set rises.
		a_NN.trainToMinimumError(trainingData,evalData,testStepSize);
		
		//Reached mininmum error, print message and run accuracy test against test set
		System.out.println("");
		System.out.println("************************************************");
		System.out.println("*** Neural Network reached training minimum! ***");
		System.out.println("***** Running against final test data set. *****");
		System.out.println("************************************************");
		System.out.println("");

		//Run against a final test set and determine accuracy based on
		//neural network output and threshold
		a_NN.runTestSet(testData,0.5);

	}
}