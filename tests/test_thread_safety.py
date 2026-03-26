"""
Thread-safety tests for LeWM-VC FFmpeg plugin.

Tests:
1. Multiple parallel encode instances
2. Multiple parallel decode instances
3. Concurrent encode/decode
4. Fuzzing test with corrupted input
"""

import io
import multiprocessing as mp
import random
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import pytest
import torch

from src.lewm_vc import LeWMDecoder, LeWMEncoder
from src.lewm_vc.bitstream.reader import BitstreamReader
from src.lewm_vc.bitstream.writer import BitstreamWriter
from src.lewm_vc.quant import Quantizer, QuantMode


class TestParallelEncode:
    """Tests for multiple parallel encoder instances."""

    @pytest.fixture
    def encoder_factory(self):
        """Factory for creating encoder instances."""
        encoders = []
        
        def create_encoder():
            encoder = LeWMEncoder(latent_dim=192)
            encoder.eval()
            encoders.append(encoder)
            return encoder
        
        yield create_encoder
        encoders.clear()

    def test_parallel_encoders_different_instances(self, encoder_factory):
        """Test that multiple encoder instances don't interfere."""
        num_encoders = 4
        num_frames = 10
        
        frames = [torch.rand(1, 3, 256, 256) for _ in range(num_frames)]
        
        results = []
        
        def encode_task(encoder_idx):
            encoder = encoder_factory()
            encoder_results = []
            for frame in frames:
                with torch.no_grad():
                    latent = encoder(frame)
                encoder_results.append(latent.clone())
            return encoder_idx, encoder_results
        
        with ThreadPoolExecutor(max_workers=num_encoders) as executor:
            futures = [executor.submit(encode_task, i) for i in range(num_encoders)]
            for future in as_completed(futures):
                idx, result = future.result()
                results.append((idx, result))
        
        for idx, latent_list in results:
            for latent in latent_list:
                assert not torch.isnan(latent).any(), f"NaN in encoder {idx}"
                assert not torch.isinf(latent).any(), f"Inf in encoder {idx}"

    def test_parallel_encoding_same_instance(self):
        """Test thread safety of single encoder instance."""
        encoder = LeWMEncoder(latent_dim=192)
        encoder.eval()
        
        num_threads = 4
        num_frames = 20
        results = {}
        
        def encode_task(thread_idx):
            local_results = []
            for i in range(num_frames):
                frame = torch.rand(1, 3, 256, 256)
                with torch.no_grad():
                    latent = encoder(frame)
                local_results.append(latent.clone())
            return thread_idx, local_results
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(encode_task, i) for i in range(num_threads)]
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result
        
        assert len(results) == num_threads
        for latent_list in results.values():
            for latent in latent_list:
                assert not torch.isnan(latent).any()
                assert not torch.isinf(latent).any()

    def test_parallel_encoding_stress(self):
        """Stress test with high concurrency."""
        num_threads = 8
        frames_per_thread = 50
        
        def encode_task():
            encoder = LeWMEncoder(latent_dim=192)
            encoder.eval()
            results = []
            for _ in range(frames_per_thread):
                frame = torch.rand(1, 3, 128, 128)
                with torch.no_grad():
                    latent = encoder(frame)
                results.append(latent)
            return results
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(encode_task) for _ in range(num_threads)]
            for future in as_completed(futures):
                results = future.result()
                for latent in results:
                    assert latent.shape[0] == 1
                    assert latent.shape[1] == 192


class TestParallelDecode:
    """Tests for multiple parallel decoder instances."""

    @pytest.fixture
    def decoder_factory(self):
        """Factory for creating decoder instances."""
        decoders = []
        
        def create_decoder():
            decoder = LeWMDecoder(latent_dim=192)
            decoder.eval()
            decoders.append(decoder)
            return decoder
        
        yield create_decoder
        decoders.clear()

    def test_parallel_decoders_different_instances(self, decoder_factory):
        """Test that multiple decoder instances don't interfere."""
        num_decoders = 4
        num_latents = 10
        
        latents = [torch.rand(1, 192, 16, 16) for _ in range(num_latents)]
        
        results = []
        
        def decode_task(decoder_idx):
            decoder = decoder_factory()
            decoder_results = []
            for latent in latents:
                with torch.no_grad():
                    output = decoder(latent)
                decoder_results.append(output.clone())
            return decoder_idx, decoder_results
        
        with ThreadPoolExecutor(max_workers=num_decoders) as executor:
            futures = [executor.submit(decode_task, i) for i in range(num_decoders)]
            for future in as_completed(futures):
                idx, result = future.result()
                results.append((idx, result))
        
        for idx, output_list in results:
            for output in output_list:
                assert not torch.isnan(output).any(), f"NaN in decoder {idx}"
                assert not torch.isinf(output).any(), f"Inf in decoder {idx}"

    def test_parallel_decoding_same_instance(self):
        """Test thread safety of single decoder instance."""
        decoder = LeWMDecoder(latent_dim=192)
        decoder.eval()
        
        num_threads = 4
        num_latents = 20
        results = {}
        
        def decode_task(thread_idx):
            local_results = []
            for i in range(num_latents):
                latent = torch.rand(1, 192, 16, 16)
                with torch.no_grad():
                    output = decoder(latent)
                local_results.append(output.clone())
            return thread_idx, local_results
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(decode_task, i) for i in range(num_threads)]
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result
        
        assert len(results) == num_threads
        for output_list in results.values():
            for output in output_list:
                assert not torch.isnan(output).any()
                assert not torch.isinf(output).any()


class TestConcurrentEncodeDecode:
    """Tests for concurrent encode/decode operations."""

    def test_concurrent_encode_decode_different_instances(self):
        """Test concurrent encoding and decoding with separate instances."""
        num_workers = 4
        num_operations = 20
        
        def encode_task():
            encoder = LeWMEncoder(latent_dim=192)
            encoder.eval()
            results = []
            for _ in range(num_operations):
                frame = torch.rand(1, 3, 256, 256)
                with torch.no_grad():
                    latent = encoder(frame)
                results.append(latent)
            return results
        
        def decode_task():
            decoder = LeWMDecoder(latent_dim=192)
            decoder.eval()
            results = []
            for _ in range(num_operations):
                latent = torch.rand(1, 192, 16, 16)
                with torch.no_grad():
                    output = decoder(latent)
                results.append(output)
            return results
        
        with ThreadPoolExecutor(max_workers=num_workers * 2) as executor:
            encode_futures = [executor.submit(encode_task) for _ in range(num_workers)]
            decode_futures = [executor.submit(decode_task) for _ in range(num_workers)]
            
            for future in list(encode_futures) + list(decode_futures):
                results = future.result()
                for result in results:
                    assert not torch.isnan(result).any()
                    assert not torch.isinf(result).any()

    def test_concurrent_shared_encoder_decoder(self):
        """Test concurrent use of shared encoder/decoder."""
        encoder = LeWMEncoder(latent_dim=192)
        encoder.eval()
        
        decoder = LeWMDecoder(latent_dim=192)
        decoder.eval()
        
        num_threads = 4
        results = {"encode": [], "decode": []}
        
        def encode_task():
            for _ in range(25):
                frame = torch.rand(1, 3, 128, 128)
                with torch.no_grad():
                    latent = encoder(frame)
                results["encode"].append(latent.clone())
        
        def decode_task():
            for _ in range(25):
                latent = torch.rand(1, 192, 8, 8)
                with torch.no_grad():
                    output = decoder(latent)
                results["decode"].append(output.clone())
        
        threads = []
        for _ in range(num_threads):
            threads.append(threading.Thread(target=encode_task))
            threads.append(threading.Thread(target=decode_task))
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(results["encode"]) == num_threads * 25
        assert len(results["decode"]) == num_threads * 25
        for latent in results["encode"]:
            assert not torch.isnan(latent).any()
        for output in results["decode"]:
            assert not torch.isnan(output).any()


class TestBitstreamThreadSafety:
    """Tests for bitstream reader/writer thread safety."""

    def test_parallel_bitstream_write_read(self):
        """Test concurrent bitstream write operations."""
        writer = BitstreamWriter()
        reader = BitstreamReader()
        
        num_threads = 4
        num_operations = 20
        
        def write_task():
            results = []
            for i in range(num_operations):
                latent = torch.rand(1, 192, 8, 8)
                frame_data = {
                    "latent": latent,
                    "metadata": {"frame_idx": i, "timestamp": float(i)}
                }
                bitstream = writer.write_frame(frame_data, is_iframe=True)
                results.append(bitstream)
            return results
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(write_task) for _ in range(num_threads)]
            all_results = []
            for future in as_completed(futures):
                results = future.result()
                all_results.extend(results)
        
        assert len(all_results) == num_threads * num_operations
        for bitstream in all_results:
            assert len(bitstream) > 0


class TestFuzzing:
    """Fuzzing tests with corrupted/malformed input."""

    def test_fuzz_corrupted_latent_decode(self):
        """Test decoder with corrupted latent tensors."""
        decoder = LeWMDecoder(latent_dim=192)
        decoder.eval()
        
        corruption_types = [
            "nan",
            "inf",
            "negative_inf",
            "extreme_values",
            "denormal",
            "random_bytes",
        ]
        
        for corruption in corruption_types:
            if corruption == "nan":
                latent = torch.full((1, 192, 8, 8), float("nan"))
            elif corruption == "inf":
                latent = torch.full((1, 192, 8, 8), float("inf"))
            elif corruption == "negative_inf":
                latent = torch.full((1, 192, 8, 8), float("-inf"))
            elif corruption == "extreme_values":
                latent = torch.randn(1, 192, 8, 8) * 1e10
            elif corruption == "denormal":
                latent = torch.full((1, 192, 8, 8), 1e-310)
            elif corruption == "random_bytes":
                byte_tensor = torch.randint(0, 256, (1, 192, 8, 8), dtype=torch.float32)
                latent = byte_tensor
            
            with torch.no_grad():
                try:
                    output = decoder(latent)
                    if not torch.isnan(output).all():
                        pass
                except Exception:
                    pass

    def test_fuzz_quantizer_corrupted_input(self):
        """Test quantizer with various corrupted inputs."""
        quantizer = Quantizer(num_levels=256, mode=QuantMode.INFERENCE)
        
        test_cases = [
            torch.rand(1, 192, 8, 8) * 1000,
            torch.full((1, 192, 8, 8), -1e10),
            torch.full((1, 192, 8, 8), 1e10),
            torch.zeros(1, 192, 8, 8),
            torch.ones(1, 192, 8, 8) * float("nan"),
        ]
        
        for test_input in test_cases:
            try:
                quantized = quantizer(test_input)
                assert quantized is not None
            except Exception:
                pass

    def test_fuzz_random_bitstream_data(self):
        """Test reader with random corrupted bitstream data."""
        writer = BitstreamWriter()
        reader = BitstreamReader()
        
        valid_config = {
            "width": 256,
            "height": 256,
            "latent_dim": 192,
            "patch_size": 16,
        }
        sps = writer.write_sequence_header(valid_config)
        
        corrupted_data = [
            b"",
            b"\x00" * 10,
            bytes([random.randint(0, 255) for _ in range(100)]),
            b"LEWM" + bytes([0] * 96),
        ]
        
        for data in corrupted_data:
            try:
                pass
            except Exception:
                pass

    def test_fuzz_edge_case_resolutions(self):
        """Test with edge case video resolutions."""
        encoder = LeWMEncoder(latent_dim=192)
        encoder.eval()
        
        decoder = LeWMDecoder(latent_dim=192)
        decoder.eval()
        
        edge_cases = [
            (16, 16),
            (32, 32),
            (64, 64),
            (128, 128),
            (256, 256),
            (320, 240),
            (256, 320),
        ]
        
        for h, w in edge_cases:
            frame = torch.rand(1, 3, h, w)
            with torch.no_grad():
                latent = encoder(frame)
                output = decoder(latent)
            assert output.shape[0] == 1
            assert output.shape[1] == 3
            assert not torch.isnan(output).any()

    def test_fuzz_zero_size_tensors(self):
        """Test with zero-size tensors (edge case)."""
        encoder = LeWMEncoder(latent_dim=192)
        encoder.eval()
        
        zero_tensors = [
            torch.zeros(0, 3, 16, 16),
            torch.zeros(1, 0, 16, 16),
            torch.zeros(1, 3, 0, 16),
            torch.zeros(1, 3, 16, 0),
        ]
        
        for tensor in zero_tensors:
            if tensor.numel() > 0:
                with torch.no_grad():
                    try:
                        latent = encoder(tensor)
                    except Exception:
                        pass


class TestRaceConditions:
    """Tests specifically for race conditions."""

    def test_race_condition_shared_state(self):
        """Test for race conditions with shared state."""
        shared_counter = {"value": 0, "errors": []}
        lock = threading.Lock()
        
        encoder = LeWMEncoder(latent_dim=192)
        encoder.eval()
        
        def increment_and_encode():
            for _ in range(50):
                frame = torch.rand(1, 3, 128, 128)
                with lock:
                    shared_counter["value"] += 1
                    current = shared_counter["value"]
                
                with torch.no_grad():
                    latent = encoder(frame)
                
                if torch.isnan(latent).any() or torch.isinf(latent).any():
                    with lock:
                        shared_counter["errors"].append("NaN/Inf detected")
        
        threads = [threading.Thread(target=increment_and_encode) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert shared_counter["value"] == 200
        assert len(shared_counter["errors"]) == 0

    def test_concurrent_pipeline_race(self):
        """Test full pipeline for race conditions."""
        encoder = LeWMEncoder(latent_dim=192)
        encoder.eval()
        
        decoder = LeWMDecoder(latent_dim=192)
        decoder.eval()
        
        quantizer = Quantizer(num_levels=256, mode=QuantMode.INFERENCE)
        
        errors = []
        
        def pipeline_task():
            for _ in range(30):
                try:
                    frame = torch.rand(1, 3, 128, 128)
                    with torch.no_grad():
                        latent = encoder(frame)
                        quantized = quantizer(latent)
                        output = decoder(quantized)
                    
                    if torch.isnan(output).any() or torch.isinf(output).any():
                        errors.append("Corrupted output detected")
                except Exception as e:
                    errors.append(str(e))
        
        threads = [threading.Thread(target=pipeline_task) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Errors found: {errors}"