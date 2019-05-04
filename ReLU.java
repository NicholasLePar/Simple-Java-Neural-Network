public class ReLU implements Neuron{
	
	int neuronType = 0;
	double[] inputs;
	double[] weights;
	
	//ReLU neuron constructor, assigns inputs and weights from parameters
	public ReLU(double[] inputs, double[] weights, double bias){
		//if the bias is 0, then there is no bias to the neuron and hence no edge weight for bias
		if(bias==0){
			this.inputs = new double[inputs.length];
			this.weights = new double[inputs.length];
			for(int i=0; i<inputs.length; i++){
				this.inputs[i] = inputs[i];
				this.weights[i] = weights[i];
			}
		}
		//However if the bias is not 0 then then there is one more weight than input,
		//this signifies the biases input. Thus we will attach the bias to the final spot in the
		//inputs array.
		else{
			this.inputs = new double[weights.length];
			this.weights = new double[weights.length];
			for(int i=0; i<inputs.length; i++){
				this.inputs[i] = inputs[i];
				this.weights[i] = weights[i];
			}
			//set the final input and weight as the bias input and weight
			this.inputs[inputs.length] = bias;
			this.weights[inputs.length] = weights[weights.length-1];
		}
	}

	//Return the bias value of the neuron, if it was anything other than 0 upon construction
	//the bias will always be the last spot in the inputs array.
	public double getBias(){
		return inputs[inputs.length-1];
	}

	//Sets the bias value of the neuron, if it was anything other than 0 upon construction
	//the bias will always be the last spot in the inputs array.
	public void setBias(double bias){
		inputs[inputs.length-1] = bias;
	}

	//return neuron type: ReLU represented as 0
	public int getNeuronType(){
		return neuronType;
	}

	//set all the inputs to all new values
	public void setInputs(double[] inputs){
		for(int i=0; i<inputs.length; i++){
			this.inputs[i] = inputs[i];
		}
	}

	//return inputs
	public double[] getInputs(){
		return inputs;
	}

	//set all the weights to new values
	public void setWeights(double[] weights){
		for(int i=0; i<weights.length; i++){
			this.weights[i] = weights[i];
		}
	}

	//return weights of inputs
	public double[] getWeights(){
		return weights;
	}

	//computes and returns the output of the neuron
	public double getOutput(){
		//u is the determined value
		double u =0;
		//for each input and edge weight collect the determined value of the neuron
		for(int i=0; i<inputs.length; i++){
			u += inputs[i]*weights[i];
		}
		//apply the function of ReLU which is max(u,0) this is the output of the neuron
		double v=0;
		if(u>=0){
			v=u;
		}
		//return the output of the neuron.
		return v;
	}

	//perform one step of stochastic gradient desecent for training one the neuron.
	//nStepSize: represents the step size (n) for adjustment in SGD
	//dEuC: refers to the partial derivative of error with respect to the intermiedate output
	//		of the above (output) layer, used in SGD to compute the dEvA of current neuron
	//outputWeight: refers to the output weight placed on the output of the ReLU going into
	//		above (output) layers as an input to another neuron
	public double[][] oneStepSGD(double nStepSize, double dEuC, double outputWeight,boolean outputLayer, double expectedValue){
		//u is the determined value
		double u =0;
		for(int i=0; i<inputs.length; i++){
			u += inputs[i]*weights[i];
		}
		//the function of ReLU is max(u,0) this is the output of the neuron
		double v=0;
		//the partial derivative of the function of ReLU (dVuA) is 1 when u>=0 else 0;
		double dVuA=0;
		if(u>=0){
			v=u;
			dVuA=1;
		}
		//partial derivative of error with respect to ReLU output is dEvA
		double dEvA;
		if(outputLayer){
			dEvA = v -expectedValue;
		}else{
			dEvA = outputWeight*dEuC;
		}
		//partial derivative of error with respect to ReLU intermediate output is dEuA
		double dEuA = dEvA*dVuA;

		//now that dEuA has been obtained calculate the partial derivative of error with 
		//respect to input edge weights (dEwA) for each input
		double dEwA [] = new double[inputs.length];
		for(int i=0; i<inputs.length; i++){
			dEwA[i] = inputs[i] * dEuA;
		}

		//save the dEuA and return it in the output as parameter 1 for future SGD
		//in lower layers as dEuC
		double [][] sgdOutput = new double[2][];
		sgdOutput[0] = new double[1];
		sgdOutput[0][0] = dEuC;
		//with dEwA for each input to ReLU neuron we can perfrom the one step of SGD
		//by calculating new edge weights for each input into the neuron
		///save the old weights and return them as SGD output for future use as 
		//output weights of lowerlayers
		sgdOutput[1] = new double[inputs.length];
		for(int i=0; i<inputs.length; i++){
			sgdOutput[1][i] = weights[i];
			weights[i] = weights[i] - nStepSize*dEwA[i];
		}
		//return the new edge weights from one step of SGD on the ReLU neuron
		return sgdOutput;
	}

	//Debug main method.
	public static void main(String[] args){
		System.out.println("ReLU Main");
		double[] testInput = {-0.2,1.7};
		double[] testWeight = {0.9,0.8,1};
		double testBias = 1;
		ReLU a_ReLU = new ReLU(testInput,testWeight,testBias);
		System.out.println(String.format("%.5f",a_ReLU.getOutput()));
	}		
}