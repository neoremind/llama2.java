import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.io.Closeable;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.Future;
import java.util.stream.IntStream;

public class Llama2 {

    static final boolean VECTOR_MATMUL_ENABLED = Boolean.parseBoolean(System.getProperty("vector.matmul.enabled", "true"));

    // 0: OFF, 1: ERROR, 2: INFO, 3: DEBUG
    static final int LOG_LEVEL = Integer.parseInt(System.getProperty("log.level", "2"));

    static final DateFormat DATE_FORMAT = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS");

    static final VectorSpecies<Float> PREF_FLOAT_SPECIES = FloatVector.SPECIES_PREFERRED;

    static final Comparator<ProbIndex> PROB_INDEX_COMPARATOR = (a, b) -> {
        if (a.prob > b.prob) return -1;
        if (a.prob < b.prob) return 1;
        return 0;
    };

    static void build_transformer(Transformer t, Config config, TransformerWeights weights, String checkpoint_path) {
        t.config = config;
        t.weights = weights;
        t.state = new RunState();
        // read in the Config and the Weights from the checkpoint
        read_checkpoint(checkpoint_path, t.config, weights);
        // allocate the RunState buffers
        malloc_run_state(t.state, t.config);
    }

    static void read_checkpoint(String checkpoint_path, Config config, TransformerWeights weights) {
        try (FileChannel file = new FileInputStream(checkpoint_path).getChannel()) {
            long file_size = file.size();
            logDebug("checkpoint_path=" + checkpoint_path + ", checkpoint_file_size_in_bytes=" + file_size);
            ByteBuffer buffer = file.map(FileChannel.MapMode.READ_ONLY, 0, Config.CONFIG_SEGMENT_BYTES);
            buffer.order(ByteOrder.LITTLE_ENDIAN);
            read_config(config, buffer);
            int shared_weights = config.vocab_size > 0 ? 1 : 0;
            config.vocab_size = Math.abs(config.vocab_size);
            logInfo(config.toString());
            long start = System.currentTimeMillis();
            logInfo("Start loading weights from " + checkpoint_path);
            // memory map the Transformer weights into the data pointer
            memory_map_weights(weights, config, checkpoint_path, shared_weights);
            logInfo("Loading weights done (elapsed " + (System.currentTimeMillis() - start) + "ms)");
        } catch (IOException e) {
            throw new RuntimeException("couldn't load " + checkpoint_path, e);
        }
    }

    static void malloc_run_state(RunState s, Config p) {
        int kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
        s.x = new float[p.dim]; // activation at current time stamp (dim,)
        s.xb = new float[p.dim]; // same, but inside a residual branch (dim,)
        s.xb2 = new float[p.dim]; // an additional buffer just for convenience (dim,)
        s.hb = new float[p.hidden_dim]; // buffer for hidden dimension in the ffn (hidden_dim,)
        s.hb2 = new float[p.hidden_dim]; // buffer for hidden dimension in the ffn (hidden_dim,)
        s.q = new float[p.dim]; // query (dim,)
        s.k = new float[kv_dim]; // query (dim,)
        s.v = new float[kv_dim]; // query (dim,)
        s.key_cache = new float[p.n_layers * p.seq_len * kv_dim]; // (layer, seq_len, dim)
        s.value_cache = new float[p.n_layers * p.seq_len * kv_dim]; // (layer, seq_len, dim)
        s.att = new float[p.n_heads * p.seq_len]; // buffer for scores/attention values (n_heads, seq_len)
        s.logits = new float[p.vocab_size]; // output logits
    }

    static void read_config(Config config, ByteBuffer buffer) {
        // e.g. hexdump -C -n 28 llama2_7b-chat.bin
        // 00000000  00 10 00 00 00 2b 00 00  20 00 00 00 20 00 00 00  |.....+.. ... ...|
        // 00000010  20 00 00 00 00 83 ff ff  00 08 00 00              | ...........|
        // dim=4096, hidden_dim=11008, n_layers=32, n_heads=32, n_kv_heads=32, vocab_size=32000, seq_len=2048
        config.dim = buffer.getInt();
        config.hidden_dim = buffer.getInt();
        config.n_layers = buffer.getInt();
        config.n_heads = buffer.getInt();
        config.n_kv_heads = buffer.getInt();
        config.vocab_size = buffer.getInt();
        config.seq_len = buffer.getInt();
    }

    static void memory_map_weights(TransformerWeights w, Config p, String checkpoint_path, int shared_weights)
            throws IOException {
        int head_size = p.dim / p.n_heads;
        // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
        int n_layers = p.n_layers;
        long pos = Config.CONFIG_SEGMENT_BYTES;

        ExecutorService executor = Executors.newFixedThreadPool(16);

        int token_embedding_table_len = p.vocab_size * p.dim;
        final long token_embedding_table_pos = pos;
        Future<float[]> token_embedding_table_future = executor.submit(() ->
                buildVector("token_embedding_table", checkpoint_path, token_embedding_table_pos,
                        token_embedding_table_len));
        pos += (long) token_embedding_table_len * Float.BYTES;

        int rms_att_weight_len = n_layers * p.dim;
        final long rms_att_weight_pos = pos;
        Future<float[][]> rms_att_weight_future = executor.submit(() ->
                buildMatrix("rms_att_weight", checkpoint_path, rms_att_weight_pos,
                        n_layers, p.dim));
        pos += (long) rms_att_weight_len * Float.BYTES;

        int wq_len = n_layers * p.dim * (p.n_heads * head_size);
        final long wq_pos = pos;
        Future<float[][]> wq_future = executor.submit(() ->
                buildMatrix("wq", checkpoint_path, wq_pos,
                        n_layers, p.dim * (p.n_heads * head_size)));
        pos += (long) wq_len * Float.BYTES;

        int wk_len = n_layers * p.dim * (p.n_kv_heads * head_size);
        final long wk_pos = pos;
        Future<float[][]> wk_future = executor.submit(() ->
                buildMatrix("wk", checkpoint_path, wk_pos,
                        n_layers, p.dim * (p.n_kv_heads * head_size)));
        pos += (long) wk_len * Float.BYTES;

        int wv_len = n_layers * p.dim * (p.n_kv_heads * head_size);
        final long wv_pos = pos;
        Future<float[][]> wv_future = executor.submit(() ->
                buildMatrix("wv", checkpoint_path, wv_pos,
                        n_layers, p.dim * (p.n_kv_heads * head_size)));
        pos += (long) wv_len * Float.BYTES;

        int wo_len = n_layers * p.n_heads * (head_size * p.dim);
        final long wo_pos = pos;
        Future<float[][]> wo_future = executor.submit(() ->
                buildMatrix("wo", checkpoint_path, wo_pos,
                        n_layers, p.n_heads * (head_size * p.dim)));
        pos += (long) wo_len * Float.BYTES;

        int rms_ffn_weight_len = n_layers * p.dim;
        final long rms_ffn_weight_pos = pos;
        Future<float[][]> rms_ffn_weight_future = executor.submit(() ->
                buildMatrix("rms_ffn_weight", checkpoint_path, rms_ffn_weight_pos,
                        n_layers, p.dim));
        pos += (long) rms_ffn_weight_len * Float.BYTES;

        int w1_len = n_layers * p.dim * p.hidden_dim;
        final long w1_pos = pos;
        Future<float[][]> w1_future = executor.submit(() ->
                buildMatrix("w1", checkpoint_path, w1_pos,
                        n_layers, p.dim * p.hidden_dim));
        pos += (long) w1_len * Float.BYTES;

        int w2_len = n_layers * p.hidden_dim * p.dim;
        final long w2_pos = pos;
        Future<float[][]> w2_future = executor.submit(() ->
                buildMatrix("w2", checkpoint_path, w2_pos,
                        n_layers, p.hidden_dim * p.dim));
        pos += (long) w2_len * Float.BYTES;

        int w3_len = n_layers * p.dim * p.hidden_dim;
        final long w3_pos = pos;
        Future<float[][]> w3_future = executor.submit(() ->
                buildMatrix("w3", checkpoint_path, w3_pos,
                        n_layers, p.dim * p.hidden_dim));
        pos += (long) w3_len * Float.BYTES;

        int rms_final_weight_len = p.dim;
        final long rms_final_weight_pos = pos;
        Future<float[]> rms_final_weight_future = executor.submit(() ->
                buildVector("rms_final_weight", checkpoint_path, rms_final_weight_pos,
                        rms_final_weight_len));
        pos += (long) rms_final_weight_len * Float.BYTES;

        pos += ((long) p.seq_len * head_size / 2) * Float.BYTES; // skip what used to be freq_cis_real (for RoPE)
        pos += ((long) p.seq_len * head_size / 2) * Float.BYTES; // skip what used to be freq_cis_imag (for RoPE)

        int wcls_len = p.vocab_size * p.dim;
        final long wcls_pos = pos;
        Future<float[]> wcls_future = (shared_weights > 0) ? null :
                executor.submit(() -> buildVector("wcls", checkpoint_path, wcls_pos, wcls_len));

        try {
            w.token_embedding_table = token_embedding_table_future.get();
            w.rms_att_weight = rms_att_weight_future.get();
            w.wq = wq_future.get();
            w.wk = wk_future.get();
            w.wv = wv_future.get();
            w.wo = wo_future.get();
            w.rms_ffn_weight = rms_ffn_weight_future.get();
            w.w1 = w1_future.get();
            w.w2 = w2_future.get();
            w.w3 = w3_future.get();
            w.rms_final_weight = rms_final_weight_future.get();
            w.wcls = wcls_future == null ? w.token_embedding_table : wcls_future.get();
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException("couldn't load weights due to " + e.getMessage(), e);
        } finally {
            executor.shutdownNow();
        }
    }

    static MemoryChunkReader createMemoryChunkReader(String checkpoint_path, long pos, long len) throws IOException {
        return new MemoryChunkReader(checkpoint_path, pos, len);
    }

    static float[] buildVector(String segmentName, String checkpoint_path, long pos, int lenOfVector)
            throws IOException {
        long expectedReadBytes = (long) lenOfVector * Float.BYTES;
        try (MemoryChunkReader memoryChunkReader = createMemoryChunkReader(checkpoint_path, pos, expectedReadBytes)) {
            float[] result = buildVector(memoryChunkReader, lenOfVector);
            logDebug(String.format("Read %s, pos=%d, len=%d, %s", segmentName, pos, expectedReadBytes, memoryChunkReader.stat));
            return result;
        }
    }

    static float[] buildVector(MemoryChunkReader chunk, int lenOfVector) {
        float[] result = new float[lenOfVector];
        for (int i = 0; i < lenOfVector; i++) {
            result[i] = chunk.getFloat();
        }
        return result;
    }

    static float[][] buildMatrix(String segmentName, String checkpoint_path, long pos, int numOfVectors, int lenOfVector)
            throws IOException {
        long expectedReadBytes = (long) numOfVectors * lenOfVector * Float.BYTES;
        try (MemoryChunkReader memoryChunkReader = createMemoryChunkReader(checkpoint_path, pos, expectedReadBytes)) {
            float[][] result = buildMatrix(memoryChunkReader, numOfVectors, lenOfVector);
            logDebug(String.format("Read %s, pos=%d, len=%d, %s", segmentName, pos, expectedReadBytes, memoryChunkReader.stat));
            return result;
        }
    }

    static float[][] buildMatrix(MemoryChunkReader chunk, int numOfVectors, int len) {
        float[][] result = new float[numOfVectors][len];
        for (int i = 0; i < numOfVectors; i++) {
            result[i] = buildVector(chunk, len);
        }
        return result;
    }

    static void build_tokenizer(Tokenizer t, String tokenizer_path, int vocab_size) {
        t.vocab_size = vocab_size;
        // malloc space to hold the scores and the strings
        t.vocab = new String[vocab_size];
        t.vocab_scores = new float[vocab_size];

        for (int i = 0; i < 256; i++) {
            t.byte_pieces[i] = String.valueOf((char) i);
        }
        try (FileChannel file = new FileInputStream(tokenizer_path).getChannel()) {
            ByteBuffer buffer = file.map(FileChannel.MapMode.READ_ONLY, 0, file.size());
            buffer.order(ByteOrder.LITTLE_ENDIAN);
            t.max_token_length = buffer.getInt();
            int len;
            for (int i = 0; i < vocab_size; i++) {
                t.vocab_scores[i] = buffer.getFloat();
                len = buffer.getInt();
                byte[] vocabBytes = new byte[len];
                buffer.get(vocabBytes);
                t.vocab[i] = new String(vocabBytes);
            }
        } catch (IOException e) {
            throw new RuntimeException("couldn't load " + tokenizer_path, e);
        }

        //t.sorted_vocab = null; // initialized lazily
        t.sorted_vocab = new HashMap<>(t.vocab_size * 4 / 3);
        for (int i = 0; i < vocab_size; i++) {
            t.sorted_vocab.put(t.vocab[i], i);
        }
        logDebug("Build tokenizer successfully, vocab_size=" + vocab_size
                + ", max_token_length=" + t.max_token_length);
    }

    static String decode(Tokenizer t, int prev_token, int token) {
        String piece = t.vocab[token];
        // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
        if (prev_token == 1 && piece.charAt(0) == ' ') {
            piece = piece.substring(1);
        }

        if (piece.length() == 6
                && piece.charAt(0) == '<'
                && piece.charAt(1) == '0'
                && piece.charAt(2) == 'x'
                && piece.charAt(5) == '>') {
            int byte_val = Integer.parseInt(piece.substring(3, 5), 16);
            piece = t.byte_pieces[byte_val];
        }
        return piece;
    }

    static int str_lookup(String str, Map<String, Integer> sorted_vocab) {
        // efficiently find the perfect match for str in vocab, return its index or -1 if not found
        return sorted_vocab.getOrDefault(str, -1);
    }

    static void build_sampler(Sampler sampler, int vocab_size, float temperature, float topp, long rng_seed) {
        sampler.vocab_size = vocab_size;
        sampler.temperature = temperature;
        sampler.topp = topp;
        sampler.rng_state = rng_seed;
        // buffer only used with nucleus sampling; may not need but it's ~small
        sampler.probindex = new ProbIndex[vocab_size];
        for (int i = 0; i < vocab_size; i++) {
            sampler.probindex[i] = new ProbIndex();
        }
    }

    static int encode(Tokenizer t, String text, boolean bos, boolean eos, int[] tokens, int num_prompt_tokens) {
        // encode the string text (input) into an upper-bound preallocated tokens[] array
        // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
        if (text == null) {
            logError("cannot encode NULL text");
            System.exit(1);
        }

        // start at 0 tokens
        int n_tokens = 0;

        // add optional BOS (=1) token, if desired
        if (bos) {
            tokens[n_tokens++] = 1;
        }

        // add_dummy_prefix is true by default
        // so prepend a dummy prefix token to the input string, but only if text != ""
        // TODO: pretty sure this isn't correct in the general case but I don't have the
        // energy to read more of the sentencepiece code to figure out what it's doing
        if (!text.isEmpty()) {
            int dummy_prefix = str_lookup(" ", t.sorted_vocab);
            tokens[n_tokens++] = dummy_prefix;
        }

        // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
        // Code point ↔ UTF-8 conversion
        // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
        // U+0000	U+007F	    0xxxxxxx
        // U+0080	U+07FF	    110xxxxx	10xxxxxx
        // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
        // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

        // process the raw (UTF-8) byte sequence of the input string
        for (int i = 0, cpi; i < text.length(); i += Character.charCount(cpi)) {
            cpi = text.codePointAt(i);

            String singleCodepoint = Character.toString(cpi);

            // ok c+1 is not a continuation byte, so we've read in a full codepoint
            int id = str_lookup(singleCodepoint, t.sorted_vocab);

            if (id != -1) {
                // we found this codepoint in vocab, add it as a token
                tokens[n_tokens++] = id;
            } else {
                // byte_fallback encoding: just encode each byte as a token
                // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
                // so the individual bytes only start at index 3
                for (byte b : singleCodepoint.getBytes(StandardCharsets.UTF_8)) {
                    tokens[n_tokens++] = Byte.toUnsignedInt(b) + 3;
                }
            }
        }

        // merge the best consecutive pair each iteration, according the scores in vocab_scores
        while (true) {
            float best_score = -1e10f;
            int best_id = -1;
            int best_idx = -1;

            for (int i = 0; i < n_tokens - 1; i++) {
                // check if we can merge the pair (tokens[i], tokens[i+1])
                String str_buf = t.vocab[tokens[i]] + t.vocab[tokens[i + 1]];
                int id = str_lookup(str_buf, t.sorted_vocab);
                if (id != -1 && t.vocab_scores[id] > best_score) {
                    // this merge pair exists in vocab! record its score and position
                    best_score = t.vocab_scores[id];
                    best_id = id;
                    best_idx = i;
                }
            }

            if (best_idx == -1) {
                break; // we couldn't find any more pairs to merge, so we're done
            }

            // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
            tokens[best_idx] = best_id;
            // delete token at position best_idx+1, shift the entire sequence back 1
            for (int i = best_idx + 1; i < n_tokens - 1; i++) {
                tokens[i] = tokens[i + 1];
            }
            n_tokens--; // token length decreased
        }

        // add optional EOS (=2) token, if desired
        if (eos) {
            tokens[n_tokens++] = 2;
        }

        return n_tokens;
    }

    static void generate(Transformer transformer, Tokenizer tokenizer, Sampler sampler, String prompt, int steps) {
        if (prompt == null) {
            prompt = "";
        }

        // encode the (string) prompt into tokens sequence
        int num_prompt_tokens = 0;
        int[] prompt_tokens = new int[prompt.length() * 2 + 3]; // +3 for '\0', ?BOS, ?EOS
        num_prompt_tokens = encode(tokenizer, prompt, true, false, prompt_tokens, num_prompt_tokens);
        if (num_prompt_tokens < 1) {
            logError("something is wrong, expected at least 1 prompt token");
            System.exit(1);
        }

        // start the main loop
        long start = 0;  // used to time our code, only initialized after first iteration
        int next;        // will store the next token in the sequence
        int token = prompt_tokens[0]; // kick off with the first token in the prompt
        int pos = 0;     // position in the sequence

        while (pos < steps) {

            // forward the transformer to get logits for the next token
            float[] logits = forward(transformer, token, pos);

            // advance the state state machine
            if (pos < num_prompt_tokens - 1) {
                // if we are still processing the input prompt, force the next prompt token
                next = prompt_tokens[pos + 1];
            } else {
                // otherwise sample the next token from the logits
                next = sample(sampler, logits);
            }
            pos++;

            // data-dependent terminating condition: the BOS (=1) token delimits sequences
            if (next == 1) {
                break;
            }

            // print the token as string, decode it with the Tokenizer object
            String piece = decode(tokenizer, token, next);
            safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
            token = next;

            // init the timer here because the first iteration can be slower
            if (start == 0) {
                start = System.currentTimeMillis();
            }
        }
        System.out.println();

        // report achieved tok/s (pos-1 because the timer starts after first iteration)
        if (pos > 1) {
            long end = System.currentTimeMillis();
            System.out.println(String.format("achieved tok/s: %f\n", (pos - 1) / (double) (end - start) * 1000));
        }
    }

    static float[] forward(Transformer transformer, int token, int pos) {
        // a few convenience variables
        Config p = transformer.config;
        TransformerWeights w = transformer.weights;
        RunState s = transformer.state;
        float[] x = s.x;
        int dim = p.dim;
        int kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
        int kv_mul = p.n_heads / p.n_kv_heads; // integer multiplier of the kv sharing in multiquery
        int hidden_dim = p.hidden_dim;
        int head_size = dim / p.n_heads;

        // copy the token embedding into x
        System.arraycopy(w.token_embedding_table, token * dim, x, 0, dim);

        // forward all the layers
        for (int l = 0; l < p.n_layers; l++) {

            // attention rmsnorm
            rmsnorm(s.xb, x, w.rms_att_weight[l], dim);

            // qkv matmuls for this position
            matmul(s.q, s.xb, w.wq[l], dim, dim);
            matmul(s.k, s.xb, w.wk[l], dim, kv_dim);
            matmul(s.v, s.xb, w.wv[l], dim, kv_dim);

            // RoPE relative positional encoding: complex-valued rotate q and k in each head
            for (int i = 0; i < dim; i += 2) {
                int head_dim = i % head_size;
                float freq = (float) (1.0F / Math.pow(10000.0F, head_dim / (float) head_size));
                float val = pos * freq;
                float fcr = (float) Math.cos(val);
                float fci = (float) Math.sin(val);
                int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
                for (int v = 0; v < rotn; v++) {
                    float[] vec = v == 0 ? s.q : s.k; // the vector to rotate (query or key)
                    float v0 = vec[i];
                    float v1 = vec[i + 1];
                    vec[i] = v0 * fcr - v1 * fci;
                    vec[i + 1] = v0 * fci + v1 * fcr;
                }
            }

            int loff = l * p.seq_len * kv_dim; // kv cache layer offset for convenience
            System.arraycopy(s.k, 0, s.key_cache, loff + pos * kv_dim, kv_dim);
            System.arraycopy(s.v, 0, s.value_cache, loff + pos * kv_dim, kv_dim);

            // multihead attention. iterate over all heads
            int h;
            for (h = 0; h < p.n_heads; h++) {
                // get the query vector for this head
                int qOffset = h * head_size;
                // attention scores for this head
                int attOffset = h * p.seq_len;
                // iterate over all timesteps, including the current one
                for (int t = 0; t <= pos; t++) {
                    // get the key vector for this head and at this timestep
                    int keyCacheOffset = loff + t * kv_dim + (h / kv_mul) * head_size;
                    // calculate the attention score as the dot product of q and k
                    float score = 0.0F;
                    for (int i = 0; i < head_size; i++) {
                        score += s.q[qOffset + i] * s.key_cache[keyCacheOffset + i];
                    }
                    score /= (float) Math.sqrt(head_size);
                    // save the score to the attention buffer
                    s.att[attOffset + t] = score;
                }

                // softmax the scores to get attention weights, from 0..pos inclusively
                softmax(s.att, attOffset, pos + 1);

                // weighted sum of the values, store back into xb
                int xbOffset = h * head_size;
                Arrays.fill(s.xb, xbOffset, xbOffset + head_size, 0F);
                for (int t = 0; t <= pos; t++) {
                    // get the value vector for this head and at this timestep
                    int valueCacheOffset = loff + t * kv_dim + (h / kv_mul) * head_size;
                    // get the attention weight for this timestep
                    float a = s.att[attOffset + t];
                    // accumulate the weighted value into xb
                    for (int i = 0; i < head_size; i++) {
                        s.xb[xbOffset + i] += a * s.value_cache[valueCacheOffset + i];
                    }
                }
            }

            // final matmul to get the output of the attention
            matmul(s.xb2, s.xb, w.wo[l], dim, dim);

            // residual connection back into x
            for (int i = 0; i < dim; i++) {
                x[i] += s.xb2[i];
            }

            // ffn rmsnorm
            rmsnorm(s.xb, x, w.rms_ffn_weight[l], dim);

            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            // first calculate self.w1(x) and self.w3(x)
            matmul(s.hb, s.xb, w.w1[l], dim, p.hidden_dim);
            matmul(s.hb2, s.xb, w.w3[l], dim, p.hidden_dim);

            for (int i = 0; i < hidden_dim; i++) {
                float val = s.hb[i];
                // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
                val *= (1.0F / (1.0F + Math.exp(-val)));
                // elementwise multiply with w3(x)
                val *= s.hb2[i];
                s.hb[i] = val;
            }

            // final matmul to get the output of the ffn
            matmul(s.xb, s.hb, w.w2[l], p.hidden_dim, dim);

            // residual connection
            for (int i = 0; i < dim; i++) {
                s.x[i] += s.xb[i];
            }
        }

        // final rmsnorm
        rmsnorm(x, x, w.rms_final_weight, dim);

        // classifier into logits
        matmul(s.logits, x, w.wcls, dim, p.vocab_size);
        return s.logits;
    }

    static void rmsnorm(float[] o, float[] x, float[] weight, int size) {
        // calculate sum of squares
        float ss = 0.0F;
        for (int j = 0; j < size; j++) {
            ss += x[j] * x[j];
        }
        ss /= size;
        ss += 1e-5f;
        ss = 1.0f / (float) Math.sqrt(ss);
        // normalize and scale
        for (int j = 0; j < size; j++) {
            o[j] = weight[j] * (ss * x[j]);
        }
    }

    static void matmul(float[] xout, float[] x, float[] w, int n, int d) {
        // W (d,n) @ x (n,) -> xout (d,)
        // by far the most amount of time is spent inside this little function
        //int i;
        IntStream.range(0, d).parallel().forEach(i -> {
            if (VECTOR_MATMUL_ENABLED) {
                xout[i] = vectorizedMatmul(w, x, i * n, n);
            } else {
                float val = 0.0f;
                for (int j = 0; j < n; j++) {
                    val += w[i * n + j] * x[j];
                }
                xout[i] = val;
            }
        });
    }

    static void softmax(float[] x, int xOffset, int size) {
        // find max value (for numerical stability)
        float max_val = x[xOffset];
        for (int i = 1; i < size; i++) {
            if (x[xOffset + i] > max_val) {
                max_val = x[xOffset + i];
            }
        }
        // exp and sum
        float sum = 0.0F;
        for (int i = 0; i < size; i++) {
            x[xOffset + i] = (float) Math.exp(x[xOffset + i] - max_val);
            sum += x[xOffset + i];
        }
        // normalize
        for (int i = 0; i < size; i++) {
            x[xOffset + i] /= sum;
        }
    }

    static int sample(Sampler sampler, float[] logits) {
        // sample the token given the logits and some hyperparameters
        int next;
        if (sampler.temperature == 0.0F) {
            // greedy argmax sampling: take the token with the highest probability
            next = sample_argmax(logits, sampler.vocab_size);
        } else {
            // apply the temperature to the logits
            for (int q = 0; q < sampler.vocab_size; q++) {
                logits[q] /= sampler.temperature;
            }
            // apply softmax to the logits to get the probabilities for next token
            softmax(logits, 0, sampler.vocab_size);
            // flip a (float) coin (this is our source of entropy for sampling)
            float coin = random_f32(sampler.rng_state);
            // we sample from this distribution to get the next token
            if (sampler.topp <= 0 || sampler.topp >= 1) {
                // simply sample from the predicted probability distribution
                next = sample_mult(logits, sampler.vocab_size, coin);
            } else {
                // top-p (nucleus) sampling, clamping the least likely tokens to zero
                next = sample_topp(logits, sampler.vocab_size, sampler.topp, sampler.probindex, coin);
            }
        }
        return next;
    }

    static int sample_argmax(float[] probabilities, int n) {
        // return the index that has the highest probability
        int max_i = 0;
        float max_p = probabilities[0];
        for (int i = 1; i < n; i++) {
            if (probabilities[i] > max_p) {
                max_i = i;
                max_p = probabilities[i];
            }
        }
        return max_i;
    }

    static float random_f32(long state) { // random float32 in [0,1)
        return (random_u32(state) >>> 8) / 16777216.0F;
    }

    static int random_u32(long state) {
        // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        return (int) ((state * 0x2545F4914F6CDD1DL) >> 32);
    }

    static int sample_mult(float[] probabilities, int n, float coin) {
        // sample index from probabilities (they must sum to 1!)
        // coin is a random number in [0, 1), usually from random_f32()
        float cdf = 0.0f;
        for (int i = 0; i < n; i++) {
            cdf += probabilities[i];
            if (coin < cdf) {
                return i;
            }
        }
        return n - 1; // in case of rounding errors
    }

    static int sample_topp(float[] probabilities, int n, float topp, ProbIndex[] probindex, float coin) {
        // top-p sampling (or "nucleus sampling") samples from the smallest set of
        // tokens that exceed probability topp. This way we never sample tokens that
        // have very low probabilities and are less likely to go "off the rails".
        // coin is a random number in [0, 1), usually from random_f32()

        int n0 = 0;
        // quicksort indices in descending order of probabilities
        // values smaller than (1 - topp) / (n - 1) cannot be part of the result
        // so for efficiency we crop these out as candidates before sorting
        float cutoff = (1.0f - topp) / (n - 1);
        for (int i = 0; i < n; i++) {
            if (probabilities[i] >= cutoff) {
                probindex[n0].index = i;
                probindex[n0].prob = probabilities[i];
                n0++;
            }
        }

        Arrays.sort(probindex, 0, n0, PROB_INDEX_COMPARATOR);

        // truncate the list where cumulative probability exceeds topp
        float cumulative_prob = 0.0f;
        int last_idx = n0 - 1; // in case of rounding errors consider all elements
        for (int i = 0; i < n0; i++) {
            cumulative_prob += probindex[i].prob;
            if (cumulative_prob > topp) {
                last_idx = i;
                break; // we've exceeded topp by including last_idx
            }
        }

        // sample from the truncated list
        float r = coin * cumulative_prob;
        float cdf = 0.0f;
        for (int i = 0; i <= last_idx; i++) {
            cdf += probindex[i].prob;
            if (r < cdf) {
                return probindex[i].index;
            }
        }
        return probindex[last_idx].index; // in case of rounding errors
    }

    static void safe_printf(String piece) {
        // piece might be a raw byte token, and we only want to print printable chars or whitespace
        // because some of the other bytes can be various control codes, backspace, etc.
        if (piece == null || piece.isEmpty()) {
            return;
        }
        if (piece.length() == 1) {
            char ch = piece.charAt(0);
            if (!((32 <= ch && ch < 127) || Character.isWhitespace(ch))) {
                return;
            }
        }
        System.out.print(piece);
    }

    static float vectorizedMatmul(float[] a, float[] b, int aOffset, int bound) {
        int i = 0;
        float res = 0.0F;

        // vector loop is unrolled 4x (4 accumulators in parallel)
        FloatVector acc1 = FloatVector.zero(PREF_FLOAT_SPECIES);
        FloatVector acc2 = FloatVector.zero(PREF_FLOAT_SPECIES);
        FloatVector acc3 = FloatVector.zero(PREF_FLOAT_SPECIES);
        FloatVector acc4 = FloatVector.zero(PREF_FLOAT_SPECIES);
        int upperBound = PREF_FLOAT_SPECIES.loopBound(bound);
        for (; i < upperBound; i += 4 * PREF_FLOAT_SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(PREF_FLOAT_SPECIES, a, aOffset + i);
            FloatVector vb = FloatVector.fromArray(PREF_FLOAT_SPECIES, b, i);
            acc1 = acc1.add(va.mul(vb));
            FloatVector vc =
                    FloatVector.fromArray(PREF_FLOAT_SPECIES, a, aOffset + i + PREF_FLOAT_SPECIES.length());
            FloatVector vd =
                    FloatVector.fromArray(PREF_FLOAT_SPECIES, b, i + PREF_FLOAT_SPECIES.length());
            acc2 = acc2.add(vc.mul(vd));
            FloatVector ve =
                    FloatVector.fromArray(PREF_FLOAT_SPECIES, a, aOffset + i + 2 * PREF_FLOAT_SPECIES.length());
            FloatVector vf =
                    FloatVector.fromArray(PREF_FLOAT_SPECIES, b, i + 2 * PREF_FLOAT_SPECIES.length());
            acc3 = acc3.add(ve.mul(vf));
            FloatVector vg =
                    FloatVector.fromArray(PREF_FLOAT_SPECIES, a, aOffset + i + 3 * PREF_FLOAT_SPECIES.length());
            FloatVector vh =
                    FloatVector.fromArray(PREF_FLOAT_SPECIES, b, i + 3 * PREF_FLOAT_SPECIES.length());
            acc4 = acc4.add(vg.mul(vh));
        }
        // reduce
        FloatVector res1 = acc1.add(acc2);
        FloatVector res2 = acc3.add(acc4);
        res += res1.add(res2).reduceLanes(VectorOperators.ADD);

        for (; i < bound; i++) {
            res += a[aOffset + i] * b[i];
        }
        return res;
    }

    static void logError(String s) {
        if (LOG_LEVEL > 0) {
            System.err.println(DATE_FORMAT.format(new Date()) + " [ERROR] " + s);
        }
    }

    static void logInfo(String s) {
        if (LOG_LEVEL > 1) {
            System.out.println(DATE_FORMAT.format(new Date()) + " [INFO] " + s);
        }
    }

    static void logDebug(String s) {
        if (LOG_LEVEL > 2) {
            System.out.println(DATE_FORMAT.format(new Date()) + " [DEBUG] " + s);
        }
    }

    static void error_usage() {
        System.err.print("Usage:   java Llama2 <checkpoint> [options]\n");
        System.err.print("Example: java Llama2 model.bin -n 256 -i \"Once upon a time\"\n");
        System.err.print("Options:\n");
        System.err.print("  -t <float>  temperature in [0,inf], default 1.0\n");
        System.err.print("  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
        System.err.print("  -s <int>    random seed, default time(NULL)\n");
        System.err.print("  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
        System.err.print("  -i <string> input prompt\n");
        System.err.print("  -z <string> optional path to custom tokenizer\n");
        System.err.print("  -m <string> mode: generate|chat, default: generate\n");
        System.err.print("  -y <string> (optional) system prompt in chat mode\n");
        System.exit(1);
    }

    public static void main(String[] args) {
        // default parameters
        String checkpoint_path = null;  // e.g. out/model.bin
        String tokenizer_path = "tokenizer.bin";
        float temperature = 1.0F;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
        float topp = 0.9F;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
        int steps = 256;            // number of steps to run for
        String prompt = null;        // prompt string
        long rng_seed = 0; // seed rng with time by default
        String mode = "generate";    // generate|chat
        String system_prompt = null; // the (optional) system prompt to use in chat mode

        // poor man's C argparse so we can override the defaults above from the command line
        if (args.length >= 1) {
            checkpoint_path = args[0];
        } else {
            error_usage();
        }

        for (int i = 1; i < args.length; i += 2) {
            // do some basic validation
            if (i + 1 >= args.length) {
                error_usage();
            } // must have arg after flag
            if (args[i].charAt(0) != '-') {
                error_usage();
            } // must start with dash
            if (args[i].length() != 2) {
                error_usage();
            } // must be -x (one dash, one letter)
            // read in the args
            char option = args[i].charAt(1);
            if (option == 't') {
                temperature = Float.parseFloat(args[i + 1]);
            } else if (option == 'p') {
                topp = Float.parseFloat(args[i + 1]);
            } else if (option == 's') {
                rng_seed = Integer.parseInt(args[i + 1]);
            } else if (option == 'n') {
                steps = Integer.parseInt(args[i + 1]);
            } else if (option == 'i') {
                prompt = args[i + 1];
            } else if (option == 'z') {
                tokenizer_path = args[i + 1];
            } else if (option == 'm') {
                mode = args[i + 1];
            } else if (option == 'y') {
                system_prompt = args[i + 1];
            } else {
                error_usage();
            }
        }

        // parameter validation/overrides
        if (rng_seed <= 0) rng_seed = System.currentTimeMillis();
        if (temperature < 0.0) temperature = 0.0F;
        if (topp < 0.0 || 1.0 < topp) topp = 0.9F;
        if (steps < 0) steps = 0;

        logInfo("Inference parameters: temperature=" + temperature + ", topp=" + topp + ", steps=" + steps
                + ", mode=" + mode
                + ", matmul.parallelism=" + ForkJoinPool.getCommonPoolParallelism()
                + ", vectorized.matmul.enabled=" + VECTOR_MATMUL_ENABLED);

        // build the Transformer via the model .bin file
        Transformer transformer = new Transformer();
        Config config = new Config();
        TransformerWeights weights = new TransformerWeights();
        transformer.config = config;
        transformer.weights = weights;
        build_transformer(transformer, config, weights, checkpoint_path);
        if (steps == 0 || steps > transformer.config.seq_len) {
            steps = transformer.config.seq_len; // ovrerride to ~max length
        }

        // build the Tokenizer via the tokenizer .bin file
        Tokenizer tokenizer = new Tokenizer();
        build_tokenizer(tokenizer, tokenizer_path, transformer.config.vocab_size);

        // build the Sampler
        Sampler sampler = new Sampler();
        build_sampler(sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

        // run!
        if ("generate".equals(mode)) {
            generate(transformer, tokenizer, sampler, prompt, steps);
        } else if ("chat".equals(mode)) {
            throw new UnsupportedOperationException("chat not supported yet");
        } else {
            logError(String.format("unknown mode: %s\n", mode));
            error_usage();
        }
    }
}

class Config {
    static final int CONFIG_SEGMENT_BYTES = 7 * Integer.BYTES;
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length

    @Override
    public String toString() {
        return "Config{" +
                "dim=" + dim +
                ", hidden_dim=" + hidden_dim +
                ", n_layers=" + n_layers +
                ", n_heads=" + n_heads +
                ", n_kv_heads=" + n_kv_heads +
                ", vocab_size=" + vocab_size +
                ", seq_len=" + seq_len +
                '}';
    }
}

class TransformerWeights {
//    // token embedding table
//    float*token_embedding_table;    // (vocab_size, dim)
//    // weights for rmsnorms
//    float*rms_att_weight; // (layer, dim) rmsnorm weights
//    float*rms_ffn_weight; // (layer, dim)
//    // weights for matmuls. note dim == n_heads * head_size
//    float*wq; // (layer, dim, n_heads * head_size)
//    float*wk; // (layer, dim, n_kv_heads * head_size)
//    float*wv; // (layer, dim, n_kv_heads * head_size)
//    float*wo; // (layer, n_heads * head_size, dim)
//    // weights for ffn
//    float*w1; // (layer, hidden_dim, dim)
//    float*w2; // (layer, dim, hidden_dim)
//    float*w3; // (layer, hidden_dim, dim)
//    // final rmsnorm
//    float*rms_final_weight; // (dim,)
//    // (optional) classifier weights for the logits, on the last layer
//    float*wcls;

    // token embedding table
    float[] token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    float[][] rms_att_weight; // (layer, dim) rmsnorm weights
    float[][] rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    float[][] wq; // (layer, dim, n_heads * head_size)
    float[][] wk; // (layer, dim, n_kv_heads * head_size)
    float[][] wv; // (layer, dim, n_kv_heads * head_size)
    float[][] wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    float[][] w1; // (layer, hidden_dim, dim)
    float[][] w2; // (layer, dim, hidden_dim)
    float[][] w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float[] rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    float[] wcls;
}

class RunState {
//    // current wave of activations
//    float *x; // activation at current time stamp (dim,)
//    float *xb; // same, but inside a residual branch (dim,)
//    float *xb2; // an additional buffer just for convenience (dim,)
//    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
//    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
//    float *q; // query (dim,)
//    float *k; // key (dim,)
//    float *v; // value (dim,)
//    float *att; // buffer for scores/attention values (n_heads, seq_len)
//    float *logits; // output logits
//    // kv cache
//    float*key_cache;   // (layer, seq_len, dim)
//    float*value_cache; // (layer, seq_len, dim)

    // current wave of activations
    float[] x; // activation at current time stamp (dim,)
    float[] xb; // same, but inside a residual branch (dim,)
    float[] xb2; // an additional buffer just for convenience (dim,)
    float[] hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float[] hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float[] q; // query (dim,)
    float[] k; // key (dim,)
    float[] v; // value (dim,)
    float[] att; // buffer for scores/attention values (n_heads, seq_len)
    float[] logits; // output logits
    // kv cache
    float[] key_cache;   // (layer, seq_len, dim)
    float[] value_cache; // (layer, seq_len, dim)
}

class Transformer {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
}

class Tokenizer {
//    char** vocab;
//    float* vocab_scores;
//    TokenIndex *sorted_vocab;
//    int vocab_size;
//    unsigned int max_token_length;
//    unsigned char byte_pieces[512]; // stores all single-byte strings

    String[] vocab;
    float[] vocab_scores;
    Map<String, Integer> sorted_vocab;
    int vocab_size;
    int max_token_length;
    String[] byte_pieces = new String[256]; // stores all single-byte strings
}

class Sampler {
    int vocab_size;
    ProbIndex[] probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    long rng_state;
}

class ProbIndex {
    float prob;
    int index;
}

class MemoryChunkReader implements Closeable {

    public static final int MAX_BATCH_READ_BYTES = 4 * 1024 * 1024;

    String filePath;
    long pos;
    long len;
    FileChannel fileChannel;
    ByteBuffer buf;
    MemoryChunkReaderStat stat;

    public MemoryChunkReader(String filePath, long pos, long len) throws IOException {
        this.filePath = filePath;
        this.pos = pos;
        this.len = len;
        this.fileChannel = new FileInputStream(filePath).getChannel();
        this.fileChannel.position(pos);
        this.buf = ByteBuffer.allocateDirect((int) Math.min(len, MAX_BATCH_READ_BYTES));
        this.buf.order(ByteOrder.LITTLE_ENDIAN);
        this.stat = new MemoryChunkReaderStat();
        doRead();
    }

    private void doRead() {
        try {
            int read = fileChannel.read(buf);
            buf.flip();
            stat.loadBytes += read;
            stat.ioTimes++;
        } catch (IOException e) {
            throw new RuntimeException("Read file failed due to " + e.getMessage()
                    + ", startPos=" + pos + ", expectedReadBytes=" + len + ", actualPos" + buf.position(), e);
        }
    }

    float getFloat() {
        if (buf.position() == buf.limit()) {
            buf.clear();
            doRead();
        }
        return buf.getFloat();
    }

    @Override
    public void close() throws IOException {
        if (fileChannel != null) {
            fileChannel.close();
        }
    }
}

class MemoryChunkReaderStat {
    long loadBytes;
    int ioTimes;

    @Override
    public String toString() {
        return "Load stats: loadBytes=" + loadBytes
                + ", ioTimes=" + ioTimes;
    }
}