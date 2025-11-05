macro_rules! blake2_impl {
    (
        word: $word:ty;
        R1: $r1:expr;
        R2: $r2:expr;
        R3: $r3:expr;
        R4: $r4:expr;
        IV: $iv:expr;
        ROUNDS: $rounds:expr;
    ) => {
        use $crate::simd::Vector4 as _;

        /// Word for the current Blake2 variant.
        pub type Word = $word;

        const R1: u32 = $r1;
        const R2: u32 = $r2;
        const R3: u32 = $r3;
        const R4: u32 = $r4;
        const IV: [Word; 8] = $iv;

        /// Default number of rounds.
        pub const ROUNDS: usize = $rounds;

        type Simd4Word = $crate::simd::Simd4<Word>;

        #[inline(always)]
        fn iv0() -> Simd4Word {
            Simd4Word::new(IV[0], IV[1], IV[2], IV[3])
        }
        #[inline(always)]
        fn iv1() -> Simd4Word {
            Simd4Word::new(IV[4], IV[5], IV[6], IV[7])
        }

        /// Compute the initial state.
        ///
        /// Panics if either `key_size` or `output_size` is greater than a word.
        pub fn initial_state(
            salt: &[Word; 2],
            persona: &[Word; 2],
            key_size: usize,
            output_size: usize,
        ) -> [Word; 8] {
            assert!(key_size <= Word::BITS as usize);
            assert!(output_size <= Word::BITS as usize);

            // Build a parameter block.
            let mut p = [0; 8];
            p[0] = 0x0101_0000 ^ ((key_size as Word) << 8) ^ (output_size as Word);
            p[4..6].copy_from_slice(salt);
            p[6..8].copy_from_slice(persona);

            // XOR parameter block with IV.
            let h = [
                iv0() ^ Simd4Word::new(p[0], p[1], p[2], p[3]),
                iv1() ^ Simd4Word::new(p[4], p[5], p[6], p[7]),
            ];

            [
                h[0].0, h[0].1, h[0].2, h[0].3, h[1].0, h[1].1, h[1].2, h[1].3,
            ]
        }

        /// Compresses the `message` block into the `state` vector.
        ///
        /// The `t` argument must contain the number of bytes hashed so
        /// far including the current message. The `f0` flag must be set
        /// when compressing the last block. The `f1` flag must be set when
        /// compressing the last block of a layer, in tree-hashing mode.
        pub fn compress<const ROUNDS: usize>(
            state: [Word; 8],
            message: &[Word; 16],
            t: u64,
            f0: Word,
            f1: Word,
        ) -> [Word; 8] {
            use $crate::consts::SIGMA;

            #[cfg_attr(not(feature = "size_opt"), inline(always))]
            fn quarter_round(v: &mut [Simd4Word; 4], rd: u32, rb: u32, m: Simd4Word) {
                v[0] = v[0].wrapping_add(v[1]).wrapping_add(m.from_le());
                v[3] = (v[3] ^ v[0]).rotate_right_const(rd);
                v[2] = v[2].wrapping_add(v[3]);
                v[1] = (v[1] ^ v[2]).rotate_right_const(rb);
            }

            #[cfg_attr(not(feature = "size_opt"), inline(always))]
            fn shuffle(v: &mut [Simd4Word; 4]) {
                v[1] = v[1].shuffle_left_1();
                v[2] = v[2].shuffle_left_2();
                v[3] = v[3].shuffle_left_3();
            }

            #[cfg_attr(not(feature = "size_opt"), inline(always))]
            fn unshuffle(v: &mut [Simd4Word; 4]) {
                v[1] = v[1].shuffle_right_1();
                v[2] = v[2].shuffle_right_2();
                v[3] = v[3].shuffle_right_3();
            }

            #[cfg_attr(not(feature = "size_opt"), inline(always))]
            fn round(v: &mut [Simd4Word; 4], m: &[Word; 16], s: &[usize; 16]) {
                quarter_round(v, R1, R2, Simd4Word::gather(m, s[0], s[2], s[4], s[6]));
                quarter_round(v, R3, R4, Simd4Word::gather(m, s[1], s[3], s[5], s[7]));

                shuffle(v);
                quarter_round(v, R1, R2, Simd4Word::gather(m, s[8], s[10], s[12], s[14]));
                quarter_round(v, R3, R4, Simd4Word::gather(m, s[9], s[11], s[13], s[15]));
                unshuffle(v);
            }

            let mut h = [
                Simd4Word::new(state[0], state[1], state[2], state[3]),
                Simd4Word::new(state[4], state[5], state[6], state[7]),
            ];

            let t0 = t as Word;
            let t1 = match Word::BITS {
                64 => 0,
                32 => (t >> 32) as Word,
                _ => unreachable!(),
            };

            let mut v = [h[0], h[1], iv0(), iv1() ^ Simd4Word::new(t0, t1, f0, f1)];

            for i in 0..ROUNDS {
                round(&mut v, message, &SIGMA[i % 10]);
            }

            h[0] = h[0] ^ (v[0] ^ v[2]);
            h[1] = h[1] ^ (v[1] ^ v[3]);

            [
                h[0].0, h[0].1, h[0].2, h[0].3, h[1].0, h[1].1, h[1].2, h[1].3,
            ]
        }
    };
}

macro_rules! blake2_core_impl {
    (
        $name:ident, $alg_name:expr, $mod:ident, $bytes:ident,
        $block_size:ident, $vardoc:expr, $doc:expr,
    ) => {
        #[derive(Clone)]
        #[doc=$vardoc]
        pub struct $name {
            h: [$mod::Word; 8],
            t: u64,
            #[cfg(feature = "reset")]
            h0: [$mod::Word; 8],
        }

        impl $name {
            /// Creates a new context with the full set of sequential-mode parameters.
            pub fn new_with_params(
                salt: &[u8],
                persona: &[u8],
                key_size: usize,
                output_size: usize,
            ) -> Self {
                // The number of bytes needed to express two words.
                let length = $bytes::to_usize() / 4;
                assert!(salt.len() <= length);
                assert!(persona.len() <= length);

                // salt is two words long
                let mut salt_array = [0 as $mod::Word; 2];
                if salt.len() < length {
                    let mut padded_salt = Array::<u8, <$bytes as Div<U4>>::Output>::default();
                    for i in 0..salt.len() {
                        padded_salt[i] = salt[i];
                    }
                    salt_array[0] =
                        $mod::Word::from_le_bytes(padded_salt[0..length / 2].try_into().unwrap());
                    salt_array[1] = $mod::Word::from_le_bytes(
                        padded_salt[length / 2..padded_salt.len()]
                            .try_into()
                            .unwrap(),
                    );
                } else {
                    salt_array[0] =
                        $mod::Word::from_le_bytes(salt[0..salt.len() / 2].try_into().unwrap());
                    salt_array[1] = $mod::Word::from_le_bytes(
                        salt[salt.len() / 2..salt.len()].try_into().unwrap(),
                    );
                }

                // persona is also two words long
                let mut persona_array = [0 as $mod::Word; 2];
                if persona.len() < length {
                    let mut padded_persona = Array::<u8, <$bytes as Div<U4>>::Output>::default();
                    for i in 0..persona.len() {
                        padded_persona[i] = persona[i];
                    }
                    persona_array[0] = $mod::Word::from_le_bytes(
                        padded_persona[0..length / 2].try_into().unwrap(),
                    );
                    persona_array[1] = $mod::Word::from_le_bytes(
                        padded_persona[length / 2..padded_persona.len()]
                            .try_into()
                            .unwrap(),
                    );
                } else {
                    persona_array[0] =
                        $mod::Word::from_le_bytes(persona[0..length / 2].try_into().unwrap());
                    persona_array[1] = $mod::Word::from_le_bytes(
                        persona[length / 2..persona.len()].try_into().unwrap(),
                    );
                }

                let h = $mod::initial_state(&salt_array, &persona_array, key_size, output_size);

                $name {
                    #[cfg(feature = "reset")]
                    h0: h.clone(),
                    h,
                    t: 0,
                }
            }

            fn finalize_with_flag(
                &mut self,
                final_block: &Array<u8, $block_size>,
                flag: $mod::Word,
                out: &mut Output<Self>,
            ) {
                self.compress(final_block, !0, flag);
                out.copy_from_slice(self.h.map(|w| w.to_le()).as_bytes())
            }

            fn compress(&mut self, block: &Block<Self>, f0: $mod::Word, f1: $mod::Word) {
                let mut m: [$mod::Word; 16] = Default::default();
                let n = core::mem::size_of::<$mod::Word>();
                for (v, chunk) in m.iter_mut().zip(block.chunks_exact(n)) {
                    *v = $mod::Word::from_ne_bytes(chunk.try_into().unwrap());
                }

                self.h = $mod::compress::<{ $mod::ROUNDS }>(self.h, &m, self.t, f0, f1);
            }
        }

        impl HashMarker for $name {}

        impl BlockSizeUser for $name {
            type BlockSize = $block_size;
        }

        impl BufferKindUser for $name {
            type BufferKind = Lazy;
        }

        impl UpdateCore for $name {
            #[inline]
            fn update_blocks(&mut self, blocks: &[Block<Self>]) {
                for block in blocks {
                    self.t += block.len() as u64;
                    self.compress(block, 0, 0);
                }
            }
        }

        impl OutputSizeUser for $name {
            type OutputSize = $bytes;
        }

        impl VariableOutputCore for $name {
            const TRUNC_SIDE: TruncSide = TruncSide::Left;

            #[inline]
            fn new(output_size: usize) -> Result<Self, InvalidOutputSize> {
                if output_size > Self::OutputSize::USIZE {
                    return Err(InvalidOutputSize);
                }
                Ok(Self::new_with_params(&[], &[], 0, output_size))
            }

            #[inline]
            fn finalize_variable_core(
                &mut self,
                buffer: &mut Buffer<Self>,
                out: &mut Output<Self>,
            ) {
                self.t += buffer.get_pos() as u64;
                let block = buffer.pad_with_zeros();
                self.finalize_with_flag(&block, 0, out);
            }
        }

        #[cfg(feature = "reset")]
        impl Reset for $name {
            fn reset(&mut self) {
                self.h = self.h0;
                self.t = 0;
            }
        }

        impl AlgorithmName for $name {
            #[inline]
            fn write_alg_name(f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str($alg_name)
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(concat!(stringify!($name), " { ... }"))
            }
        }

        impl Drop for $name {
            fn drop(&mut self) {
                #[cfg(feature = "zeroize")]
                {
                    self.h.zeroize();
                    self.t.zeroize();
                }
            }
        }

        impl VariableOutputCoreCustomized for $name {
            #[inline]
            fn new_customized(customization: &[u8], output_size: usize) -> Self {
                Self::new_with_params(&[], customization, 0, output_size)
            }
        }

        #[cfg(feature = "zeroize")]
        impl ZeroizeOnDrop for $name {}
    };
}

macro_rules! blake2_mac_impl {
    (
        $name:ident, $hash:ty, $max_size:ty, $doc:expr
    ) => {
        #[derive(Clone)]
        #[doc=$doc]
        pub struct $name<OutSize>
        where
            OutSize: ArraySize + IsLessOrEqual<$max_size, Output = True>,
        {
            core: $hash,
            buffer: LazyBuffer<<$hash as BlockSizeUser>::BlockSize>,
            #[cfg(feature = "reset")]
            key_block: Option<Key<Self>>,
            _out: PhantomData<OutSize>,
        }

        impl<OutSize> $name<OutSize>
        where
            OutSize: ArraySize + IsLessOrEqual<$max_size, Output = True>,
        {
            /// Create new instance using provided key, salt, and persona.
            ///
            /// Setting key to `None` indicates unkeyed usage.
            ///
            /// # Errors
            ///
            /// If key is `Some`, then its length should not be zero or bigger
            /// than the block size. The salt and persona length should not be
            /// bigger than quarter of block size. If any of those conditions is
            /// false the method will return an error.
            #[inline]
            pub fn new_with_salt_and_personal(
                key: Option<&[u8]>,
                salt: &[u8],
                persona: &[u8],
            ) -> Result<Self, InvalidLength> {
                let kl = key.map_or(0, |k| k.len());
                let bs = <$hash as BlockSizeUser>::BlockSize::USIZE;
                let qbs = bs / 4;
                if key.is_some() && kl == 0 || kl > bs || salt.len() > qbs || persona.len() > qbs {
                    return Err(InvalidLength);
                }
                let buffer = if let Some(k) = key {
                    let mut padded_key = Block::<$hash>::default();
                    padded_key[..kl].copy_from_slice(k);
                    LazyBuffer::new(&padded_key)
                } else {
                    LazyBuffer::default()
                };
                Ok(Self {
                    core: <$hash>::new_with_params(salt, persona, kl, OutSize::USIZE),
                    buffer,
                    #[cfg(feature = "reset")]
                    key_block: key.map(|k| {
                        let mut t = Key::<Self>::default();
                        t[..kl].copy_from_slice(k);
                        t
                    }),
                    _out: PhantomData,
                })
            }
        }

        impl<OutSize> KeySizeUser for $name<OutSize>
        where
            OutSize: ArraySize + IsLessOrEqual<$max_size, Output = True>,
        {
            type KeySize = $max_size;
        }

        impl<OutSize> KeyInit for $name<OutSize>
        where
            OutSize: ArraySize + IsLessOrEqual<$max_size, Output = True>,
        {
            #[inline]
            fn new(key: &Key<Self>) -> Self {
                Self::new_from_slice(key).expect("Key has correct length")
            }

            #[inline]
            fn new_from_slice(key: &[u8]) -> Result<Self, InvalidLength> {
                let kl = key.len();
                if kl > <Self as KeySizeUser>::KeySize::USIZE {
                    return Err(InvalidLength);
                }
                let mut padded_key = Block::<$hash>::default();
                padded_key[..kl].copy_from_slice(key);
                Ok(Self {
                    core: <$hash>::new_with_params(&[], &[], key.len(), OutSize::USIZE),
                    buffer: LazyBuffer::new(&padded_key),
                    #[cfg(feature = "reset")]
                    key_block: {
                        let mut t = Key::<Self>::default();
                        t[..kl].copy_from_slice(key);
                        Some(t)
                    },
                    _out: PhantomData,
                })
            }
        }

        impl<OutSize> Update for $name<OutSize>
        where
            OutSize: ArraySize + IsLessOrEqual<$max_size, Output = True>,
        {
            #[inline]
            fn update(&mut self, input: &[u8]) {
                let Self { core, buffer, .. } = self;
                buffer.digest_blocks(input, |blocks| core.update_blocks(blocks));
            }
        }

        impl<OutSize> OutputSizeUser for $name<OutSize>
        where
            OutSize: ArraySize + IsLessOrEqual<$max_size, Output = True>,
        {
            type OutputSize = OutSize;
        }

        impl<OutSize> FixedOutput for $name<OutSize>
        where
            OutSize: ArraySize + IsLessOrEqual<$max_size, Output = True>,
        {
            #[inline]
            fn finalize_into(mut self, out: &mut Output<Self>) {
                let Self { core, buffer, .. } = &mut self;
                let mut full_res = Default::default();
                core.finalize_variable_core(buffer, &mut full_res);
                out.copy_from_slice(&full_res[..OutSize::USIZE]);
            }
        }

        #[cfg(feature = "reset")]
        impl<OutSize> Reset for $name<OutSize>
        where
            OutSize: ArraySize + IsLessOrEqual<$max_size, Output = True>,
        {
            fn reset(&mut self) {
                self.core.reset();
                self.buffer = if let Some(k) = self.key_block {
                    let kl = k.len();
                    let mut padded_key = Block::<$hash>::default();
                    padded_key[..kl].copy_from_slice(&k);
                    LazyBuffer::new(&padded_key)
                } else {
                    LazyBuffer::default()
                }
            }
        }

        #[cfg(feature = "reset")]
        impl<OutSize> FixedOutputReset for $name<OutSize>
        where
            OutSize: ArraySize + IsLessOrEqual<$max_size, Output = True>,
        {
            #[inline]
            fn finalize_into_reset(&mut self, out: &mut Output<Self>) {
                let Self { core, buffer, .. } = self;
                let mut full_res = Default::default();
                core.finalize_variable_core(buffer, &mut full_res);
                out.copy_from_slice(&full_res[..OutSize::USIZE]);
                self.reset();
            }
        }

        impl<OutSize> MacMarker for $name<OutSize> where
            OutSize: ArraySize + IsLessOrEqual<$max_size, Output = True>
        {
        }

        impl<OutSize> fmt::Debug for $name<OutSize>
        where
            OutSize: ArraySize + IsLessOrEqual<$max_size, Output = True>,
        {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}{} {{ ... }}", stringify!($name), OutSize::USIZE)
            }
        }

        impl<OutSize> Drop for $name<OutSize>
        where
            OutSize: ArraySize + IsLessOrEqual<$max_size, Output = True>,
        {
            fn drop(&mut self) {
                #[cfg(feature = "zeroize")]
                {
                    // `self.core` zeroized by its `Drop` impl
                    self.buffer.zeroize();
                    #[cfg(feature = "reset")]
                    if let Some(mut key_block) = self.key_block {
                        key_block.zeroize();
                    }
                }
            }
        }
        #[cfg(feature = "zeroize")]
        impl<OutSize> ZeroizeOnDrop for $name<OutSize> where
            OutSize: ArraySize + IsLessOrEqual<$max_size, Output = True>
        {
        }
    };
}
