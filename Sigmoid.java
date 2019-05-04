import java.math.*;

public class Sigmoid implements Neuron{
	
	int neuronType = 1;
	double[] inputs;
	double[] weights;
	
	//Sigmoid neuron constructor, assigns inputs and weights from parameters
	public Sigmoid(double[] inputs, double[] weights, double bias){
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

	//return neuron type: Sigmoid represented as 1
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
		for(int i=0; i<inputs.length; i++){
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
		//apply the function of Sigmoid which is 1/(1+e^(-z)) this is the output of the neuron
		double v = 1/(Math.exp(-u)+1);

		//return the output of the neuron.
		return v;
	}

	//perform one step of stochastic gradient desecent for training one the neuron.
	//return two outputs:
	//		1.the partial derivative of error with respect to the
	//			intermideate layer for future use of lower layer SGD.
	//		2.the old weights for reference in SGD to lower layer output weight
	//
	//Parameters: 
	//nStepSize: represents the step size (n) for adjustment in SGD
	//dEuC: refers to the partial derivative of error with respect to the intermiedate output
	//		of the above (output) layer, used in SGD to compute the dEvA of current neuron
	//outputWeight: refers to the output weight placed on the output of the Sigmoid going into
	//		above (output) layers as an input to another neuron
	public double[][] oneStepSGD(double nStepSize, double dEuC, double outputWeight, boolean outputLayer, double expectedValue){
		//u is the determined value
		double u =0;
		for(int i=0; i<inputs.length; i++){
			u += inputs[i]*weights[i];
		}
		//the function of Sigmoid is 1/(1+e^(-z)) this is the output of the neuron
		double v = 1/(Math.exp(-u)+1);
		//the partial derivative of the function of Sigmoid is e^-x / (1 + e^-x)^2
		double dVuA= Math.exp(-u)/Math.pow( (Math.exp(-u)+1) , 2 );
		//partial derivative of error with respect to Sigmoid output is dEvA
		double dEvA;
		if(outputLayer){
			dEvA = v - expectedValue;
		}
		else{
			dEvA = outputWeight*dEuC;	
		} 
		//partial derivative of error with respect to Sigmoid intermediate output is dEuA
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
		sgdOutput[0][0] = dEuA;
		//with dEwA for each input to Sigmoid neuron we can perfrom the one step of SGD
		//by calculating new edge weights for each input into the neuron.
		///save the old weights and return them as SGD output for future use as 
		//output weights of lowerlayers
		sgdOutput[1] = new double[inputs.length];
		for(int i=0; i<inputs.length; i++){
			sgdOutput[1][i] = weights[i];
			weights[i] = weights[i] - nStepSize*dEwA[i];
		}

		//return the new edge weights from one step of SGD on the Sigmoid neuron
		return sgdOutput;
	}

	//Debug main method
	public static void main(String[] args){
		System.out.println("Sigmoid Main");
		double[] testInput = {2.18,1.43};
		double[] testWeight = {0.3,0.2,0.4};
		double testBias = 1;
		Sigmoid a_Sigmoid = new Sigmoid(testInput,testWeight,testBias);
		System.out.println(String.format("%.5f",a_Sigmoid.getOutput()));
	}		
}