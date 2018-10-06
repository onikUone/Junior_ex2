import java.util.Arrays;

public class Neuron {
	//member field
	protected double[] weight;
	protected double threshoud;
	protected double net;
	protected double eta;
	protected double alpha;
	protected double[] delta_W;
	protected double delta_T;

	public double[] getWeight(){
		return weight;
	}

	public void forward_function(Neuron preNeuron[]) {
		net = 0.0;
		for(int i=0; i<this.weight.length; i++) {
			net += weight[i] * preNeuron[i].output();
		}
		net += threshoud;
	}

	public double output() {	//入力層のみオーバーライドする
		return 1.0/(1.0 + Math.exp(-net));
	}

	//constructor
	Neuron(int preNumber, double preWeight, double preThreshoud, double preEta, double preAlpha) {
		weight = new double[preNumber];
		for (int i = 0; i < preNumber; i++) {
			weight[i] = preWeight;
		}
		threshoud = preThreshoud;
		eta = preEta;
		alpha = preAlpha;
		delta_W = new double[weight.length];
		Arrays.fill(delta_W, 0.0);
		delta_T = 0.0;
	}
	Neuron(){	//入力層の時はオーバーライドして何もしない

	}
}
