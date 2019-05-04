public interface Neuron extends java.io.Serializable  {
	public int getNeuronType();
	public void setBias(double bias);
	public double getBias();
	public void setInputs(double[] inputs);
	public double[] getInputs();
	public void setWeights(double[] weights);
	public double[] getWeights();
	public double getOutput();
	public double[][] oneStepSGD(double nStepSize, double dEuC, double outputWeight,boolean outputLayer, double expectedValue);
}