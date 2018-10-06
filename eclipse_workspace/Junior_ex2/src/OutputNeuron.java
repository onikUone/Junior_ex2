public class OutputNeuron extends Neuron{
	//member field

	//member method
//	public void forward_function(InterNeuron inter[]) {
//		net = 0.0;
//		for(int i=0; i<this.weight.length; i++) {
//			net += weight[i] * inter[i].output();
//		}
//		net += threshoud;
//	}
	public void reWeight(InterNeuron inter[], double y) {
		for(int i=0; i<this.weight.length; i++) {
			delta_W[i] *= alpha;
			delta_W[i] += eta * (y - this.output()) * this.output() * (1 - this.output()) * inter[i].output();
			weight[i] += delta_W[i];	//重み更新
		}
		delta_T *= alpha;
		delta_T += eta * (y - this.output()) * this.output() * (1 - this.output());
		threshoud += delta_T;	//しきい値更新
	}

	//constructor
	OutputNeuron(int interNumber, double preWeight, double preThreshoud, double preEta, double preAlpha) {
		super(interNumber, preWeight, preThreshoud, preEta, preAlpha);
	}
}
