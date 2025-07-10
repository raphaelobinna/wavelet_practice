import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
import pywt
from scipy import signal
import pandas as pd


class ECGWaveletDecomposer:
    def __init__(self, wavelet='db4', max_level=6):
        """
        Initialize ECG Wavelet Packet Decomposer
        
        Parameters:
        - wavelet: Wavelet type (db4, db6, coif2, biorthogonal wavelets work well for ECG)
        - max_level: Maximum decomposition level
        """
        self.wavelet = wavelet
        self.max_level = max_level
        self.wp_tree = None
        self.reconstructed_bands = {}


    def decompose_ecg(self, ecg_signal, sampling_rate=1000):
        self.signal = ecg_signal
        self.sampling_rate = sampling_rate

        decomposed = pywt.WaveletPacket(ecg_signal, self.wavelet, maxlevel=self.max_level)
        self.wp_tree =decomposed
        return decomposed
    
    def get_frequency_bands(self):
        """
        Calculate frequency ranges for each decomposition level
        Returns dictionary with band names and frequency ranges
        """
        nyquist = self.sampling_rate / 2
        bands = {}
        
        for level in range(1, self.max_level + 1):
            num_bands = 2 ** level
            freq_step = nyquist / num_bands
            
            for i in range(num_bands):
                band_name = f"level_{level}_band_{i}"
                freq_low = i * freq_step
                freq_high = (i + 1) * freq_step
                bands[band_name] = (freq_low, freq_high)
        
        return bands
    
    def reconstruct_frequency_bands(self, target_level=4):

        reconstructed = {}

        nodes =[node.path  for node in self.wp_tree.get_level(target_level, 'freq')]
        for node_path in nodes:
            # Wavelet Packet reconstruction using best basis nodes
            
            wp_copy = pywt.WaveletPacket(data=self.signal, wavelet=self.wavelet, maxlevel=self.max_level)
            for level in range(target_level+1):
                level_nodes = [node.path for node in wp_copy.get_level(level, 'freq')]
                for path in level_nodes:
                    if path != node_path and not node_path.startswith(path):
                        wp_copy[path].data = np.zeros_like(wp_copy[path].data)

            reconstructed_signal = wp_copy.reconstruct(update=True)

            if len(reconstructed_signal) != len(self.signal):
                if len(reconstructed_signal) > len(self.signal):
                    reconstructed_signal = reconstructed_signal[:len(self.signal)]
                else:
                    reconstructed_signal = np.pad(reconstructed_signal, (0, len(self.ecg_signal) - len(reconstructed_signal)),mode='constant')

            reconstructed[node_path] = reconstructed_signal

        self.reconstructed_bands = reconstructed
        return reconstructed

    def reconstruct_frequency_bands_method2(self, target_level=4):
        nodes = [node.path for node in self.wp_tree.get_level(target_level)]

        reconstructed={}

        for node_path in nodes:
            wp_copy = pywt.WaveletPacket(self.signal, self.wavelet, maxlevel=self.max_level)

            wp_copy[node_path] = self.wp_tree[node_path].data

            reconstructed_signal = wp_copy.reconstruct(update=True)

            if len(reconstructed_signal) != len(self.signal):
                if len(reconstructed_signal) > len(self.signal):
                    reconstructed_signal = reconstructed_signal[:len(self.signal)]
                else:
                    reconstructed_signal = np.pad(reconstructed_signal, (0, len(self.ecg_signal) - len(reconstructed_signal)),mode='constant')

            reconstructed[node_path] = reconstructed_signal

        self.reconstructed_bands = reconstructed
        return reconstructed

    def map_node_path_to_frequency(self, node_path, target_level):
        """
        Map a node path to its corresponding frequency band
        
        Parameters:
        - node_path: Binary string representing the path (e.g., '0101')
        - target_level: The decomposition level
        
        Returns:
        - Tuple of (freq_low, freq_high) in Hz
        """
        binary_path = node_path.replace('a', '0').replace('d', '1')
        # Convert binary path to band index
        band_index = int(binary_path, 2) if node_path else 0
        
        # Calculate frequency range
        nyquist = self.sampling_rate / 2
        num_bands = 2 ** target_level
        freq_step = nyquist / num_bands
        
        freq_low = band_index * freq_step
        freq_high = (band_index + 1) * freq_step
        
        return freq_low, freq_high
    
    def plot_only_qrs(self, target_level):
        self.max_level = 6
        self.wp_tree = pywt.WaveletPacket(data=self.signal, wavelet=self.wavelet, maxlevel=self.max_level)
        all_node_paths = [node.path for node in self.wp_tree.get_level(target_level, 'freq')]
        qrs_nodes = []
        for node_path in all_node_paths:
            low_f, high_f = self.map_node_path_to_frequency(node_path, target_level)

            if low_f >= 5 and high_f <= 50:
                qrs_nodes.append(node_path)

        print('gotten nodes', qrs_nodes)
        
        new_reconstruct = pywt.WaveletPacket(data=None, wavelet=self.wavelet, maxlevel=self.max_level)

        for path in qrs_nodes:
            new_reconstruct[path] = self.wp_tree[path].data

        qrs_signal = new_reconstruct.reconstruct()



        plt.figure(figsize=(12, 5))
        plt.plot(self.signal, label="Original ECG", alpha=0.5)
        plt.plot(qrs_signal, label="QRS-only Reconstructed", linewidth=2)
        plt.legend()
        plt.title("QRS Component Isolated Using Wavelet Packets (5–40 Hz)")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.tight_layout()
        plt.show()



    def demonstrate_reconstruction_methods(self, target_level=4):
        """
        Compare different reconstruction methods
        """
        print("Comparing reconstruction methods...")
        
        # Method 1: Zeroing out other nodes
        print("Method 1: Zeroing out other nodes...")
        reconstructed1 = self.reconstruct_frequency_bands(target_level)
        
        # Method 3: Selective reconstruction (most reliable)
        print("Method 3: Selective reconstruction...")
        reconstructed3 = self.reconstruct_frequency_bands_method2(target_level)
        
        # Compare results
        print(f"\nReconstructed {len(reconstructed1)} bands using Method 1")
        print(f"Reconstructed {len(reconstructed3)} bands using Method 3")

        # Plot comparison for first few bands
        time = np.arange(len(self.signal)) / self.sampling_rate
        
        fig, axes = plt.subplots(min(4, len(reconstructed3)), 1, figsize=(15, 10))
        if len(reconstructed3) == 1:
            axes = [axes]
        
        node_paths = list(reconstructed3.keys())[:min(4, len(reconstructed3))]
        
        for i, node_path in enumerate(node_paths):
            freq_low, freq_high = self.map_node_path_to_frequency(node_path, target_level)
            
            axes[i].plot(time, reconstructed1[node_path], 'b-', alpha=0.7, label='Method 1')
            axes[i].plot(time, reconstructed3[node_path], 'r--', alpha=0.7, label='Method 3')
            axes[i].set_title(f'Node {node_path} ({freq_low:.1f}-{freq_high:.1f} Hz)')
            axes[i].set_ylabel('Amplitude')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        plt.show()
        
        # the most reliable method (Method 3) for subsequent analysis
        self.reconstructed_bands = reconstructed3
        return reconstructed3
    

def demonstrate_reconstruction_understanding():
    """
    Demonstrate the reconstruction process step by step
    """
    print("=== Understanding Wavelet Packet Reconstruction ===\n")
    
    # Generate ECG signal
    ecg_signal = nk.ecg_simulate(duration=5, sampling_rate=1000, heart_rate=70)
    
    # Initialize decomposer
    decomposer = ECGWaveletDecomposer(wavelet='db4', max_level=4)
    
    # Decompose
    wp_tree = decomposer.decompose_ecg(ecg_signal, sampling_rate=1000)
    
    # Show the tree structure
    print("Wavelet Packet Tree Structure:")
    print("Level 0 (root):", wp_tree.path, "- Full signal")
    print("Level 1:", [node.path for node in wp_tree.get_level(1, 'freq')])
    print("Level 2:", [node.path for node in wp_tree.get_level(2, 'freq')])
    print("Level 3:", [node.path for node in wp_tree.get_level(3, 'freq')])
    print("Level 4:", [node.path for node in wp_tree.get_level(4, 'freq')])
    
    # Show frequency mapping
    print("\nFrequency Mapping (Level 4):")
    nodes_level4 = [node.path for node in wp_tree.get_level(4, 'freq')]
    for node_path in nodes_level4:
        freq_low, freq_high = decomposer.map_node_path_to_frequency(node_path, 4)
        print(f"Node '{node_path}' -> {freq_low:.1f}-{freq_high:.1f} Hz")
    
    # Demonstrate reconstruction
    print("\n=== Reconstruction Process ===")
    reconstructed = decomposer.demonstrate_reconstruction_methods(target_level=4)

    decomposer.plot_only_qrs(target_level=6)
    
    # Show energy distribution
    print("\nEnergy Distribution:")
    for node_path, signal in reconstructed.items():
        energy = np.sum(signal**2)
        freq_low, freq_high = decomposer.map_node_path_to_frequency(node_path, 4)
        print(f"Node '{node_path}' ({freq_low:.1f}-{freq_high:.1f} Hz): Energy = {energy:.2f}")
    
    return decomposer, reconstructed

if __name__ == "__main__":
    # Run the demonstration
    decomposer, reconstructed = demonstrate_reconstruction_understanding()
    
    print("\n=== Key Takeaways ===")
    print("1. Node paths are binary strings representing the decomposition tree")
    print("2. Each path corresponds to a specific frequency band")
    print("3. Reconstruction requires careful handling of tree structure")
    print("4. Method 3 (selective reconstruction) is most reliable")
    print("5. Always verify reconstructed signal lengths match original")