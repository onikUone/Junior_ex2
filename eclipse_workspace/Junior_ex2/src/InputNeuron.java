public class InputNeuron extends Neuron {
	//member field
	double x;

	//member method
	public void input(double xData) {
		x = xData;
	}
	public double output() {
		return x;
	}

	//constructor
	InputNeuron(){
		super();
	}
}
