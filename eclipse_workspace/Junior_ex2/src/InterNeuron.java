public class InterNeuron extends Neuron{
	//member field

	//member method
	public void reWeight(double x, double y, double o_fromInter, OutputNeuron out, int inputNumber){
		double temp = 0.0;
		temp += (y - out.output()) * out.output() * (1 - out.output()) * out.getWeight()[inputNumber];
		for(int i=0; i<weight.length; i++){
			delta_W[i] *= alpha;
			delta_W[i] += eta * o_fromInter * (1 - o_fromInter) * temp * x;
			weight[i] += delta_W[i];
		}
		delta_T *= alpha;
		delta_T += eta * o_fromInter * (1 - o_fromInter) * temp;
		threshoud += delta_T;
	}

	//constructor
	InterNeuron(int inputNumber, double preWeight, double preThreshoud, double preEta, double preAlpha) {
		super(inputNumber, preWeight, preThreshoud, preEta, preAlpha);
	}
}
