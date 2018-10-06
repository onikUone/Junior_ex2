public class InterNeuron extends Neuron{
	//member field

	//member method
//	public void forward_function(InputNeuron input[]) {
//		net = 0.0;
//		for(int i=0; i<this.weight.length; i++) {
//			net += weight[i] * input[i].output();
//		}
//		net += threshoud;
//	}

	public void reWeight(InputNeuron input[], OutputNeuron out[] , double[] y, int myIndex) {
		double temp = 0.0;
		for(int i=0 ; i<out.length; i++){
			temp += (y[i] - out[i].output()) * out[i].output() * (1 - out[i].output()) * out[i].getWeight()[myIndex];
		}
		for(int i=0; i<this.weight.length; i++	) {
			delta_W[i] *= alpha;
			delta_W[i] += eta * this.output() * (1 - this.output()) * temp * input[i].output();
			weight[i] += delta_W[i];	//重み更新
		}
		delta_T *= alpha;
		delta_T += eta * this.output() *(1 - this.output()) * temp;
		threshoud += delta_T;	//しきい値更新
	}

	//constructor
	InterNeuron(int inputNumber, double preWeight, double preThreshoud, double preEta, double preAlpha) {
		super(inputNumber, preWeight, preThreshoud, preEta, preAlpha);
	}
}
