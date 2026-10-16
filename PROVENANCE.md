# Provenance

This file lists the SHA-256 of every file in the repository next to the SHA-256 of the file as it was run,
and states what differs between the two. `SHA256SUMS` carries the published digests in the format read by
`sha256sum -c`.

## What differs from the files as run

- **Python sources.** Comments were removed; docstrings were removed and a short English module docstring was
  added; string literals of four or more words (log and console messages, help texts, report templates) were
  translated into English. The absolute data-root prefix hard-coded in the scripts (a Google Drive folder on
  Colab or a local drive letter) was replaced by the variable `INTERNAL_DATA_ROOT`, read from the environment
  variable `ARPIA_ROOT` in one statement added at the top of each such script, and in three output paths a project-specific
  folder name was replaced by the generic name `internal`. The executable code is otherwise unchanged:
  `verify_code.py` recomputes, for each script, the digest of its syntax tree with those parts normalized and
  compares it with the digest of the script as run, listed in `CODE_AST_SHA256` and in the last column below.
- **YAML configuration files.** Comments were removed, the data-root prefix was replaced by `${ARPIA_ROOT}`
  (or `${ARPIA_V2_ROOT}`), and descriptive values of four or more words were translated into English;
  `verify_code.py` checks the parsed content in the same way.

Because the bytes of the scripts changed, a digest of a script recorded elsewhere refers to the file as run;
the second column below gives those.

## Every file

| file | sha256 as run | sha256 published here | check |
| --- | --- | --- | --- |
| `.gitignore` | n/a (written for this package) | `a0d92e2b71a221d63a253a681f814ff986667aa738a3693e2ac312cd34d3d86d` | new file |
| `CITATION.cff` | n/a (written for this package) | `30dc56d90311dea1c57830945a9022ae51234c787fb4bf03aa48d67cc6b0a92c` | new file |
| `LICENSE` | n/a (written for this package) | `c7f74b894abf59b258d0a4453cd8b726182182955e1ee29a8216e9139408d4bf` | new file |
| `README.md` | n/a (written for this package) | `742be19dac3a25188d51cd613dcd3f0d33e2a8041b92e7e02227faa8b1b4f86a` | new file |
| `analysis/computational_cost/computational_cost_colab_v2.py` | `1593a4b876fd6905316dbec89ef75a94d4f881506e567843b14d0b87a730c5e9` | `ef97eeff2860fd5fa57657bb1e2d3dbb0f46194d2734c1472f25045202b671c2` | code digest `e4d554092b64caf7c137e55998f51b38ee538bdd9f1b03fcd9b7f831d1290409` |
| `analysis/computational_cost/computational_cost_local_v2.py` | `36fbb4c32a305c01e56db3e24a9b7022df9a7dbf1d3a28a9959903cc1928c5cb` | `979b04d748881252e5d8a5a1ab699afd9e823d000b7fd7a8461438ed8cfac18a` | code digest `27e33dc36db2f8e8c0a5f56a98a4b0d9dabee010ce5a9333e28926ddcf6597de` |
| `analysis/computational_cost/generate_figures_v2.py` | `d66351fc0e11090df1659aaade2420b4aa6f7f9348c3cd9589a84826ddc75a84` | `7de920cf2cf0644f6dd98c556e055a80ed7edd32b7e8edde49d1a56826979cbc` | code digest `75986284a104c1214dae5f3a6cc28e63ac77f79b575e467db64c9a7180f99630` |
| `baselines/classical/run_classical_methods_enhanced.py` | `902b413fe51c50b6f1e6804f5cd08de8851f61ba443ea8535638fbf1951e095b` | `eaabb359165e6ba6388ac92bbc9206a78ec1364f154439ecf324cf02e950e177` | code digest `6de25abe14966ea27ad5f74450b3c87ba8362d8ba58795cbac5a44d94f8b16d6` |
| `baselines/classical/run_classical_methods_rigorous.py` | `192ae1cea2943c3242633f8b43cfd62e8de112ec19adba6996334a85c9315221` | `9f5b92110dd63e9bde1857263b89da6104f5bfcfe5c4f9273d18713f37a918ef` | code digest `475fd54b0e723030af9d7c01a4aa89dd48c3644844d5d1a6167fb7a5d07c6ecb` |
| `enrichment/edges/arestas_config_sequencial.yaml` | `eed6cadb47d8d537be2f2ed870d82cdb2a5dffb76e5794eefa2226081f552249` | `b7db2d2ac4abc058f3a7aeb20a6996970e411884a78c9aef02156cd7d17d9a15` | code digest `eddde1961f4db802ec868ee5caf15faeeea2dbdf9b13962a523d01d06457983b` |
| `enrichment/edges/gerador_arestas_v3_sequencial.py` | `ed16da7903078cb8db445a354e71ff1d2bba48ac29330e6d7627d0c0bc04a295` | `4ba41da82a42e260c86f8290d0eca6e7356dabca5a009436a6f9207189ba93e3` | code digest `807edfde5244ecfe16b2d282f40aada9a43a1f95a52c94bf8e1d90bcc3db535e` |
| `enrichment/edges/run_arestas_sequencial.py` | `ee79c7b0efc8a3b556640ae54996fa842b5ce12ac69986a219fc062b2998ee00` | `30f7005201aeb25be540fd455990ed68d0428a38c27e64d31552320adce5076e` | code digest `af7bf337912fdac00587eed257df518d195c2d0e385cb275f0f9b3c42a181934` |
| `enrichment/edges/validador_arestas_v3.py` | `3f8a54b434260a45859128035052f6e350ea1a59db62f2b97c0030ea775bbe13` | `a0b25dab0f0a4792aac5469c0826a60864e13761a72a44a1fd75a2f917bcd321` | code digest `751981ddf93cea8b99171dbdcb7fc356a563e9c76a9be3909e529af6477c04d3` |
| `enrichment/target/alvo_config.yaml` | `5a8fe49a8ce0163fcf24fbcffc50d43b5907b9415a93b7b8fbb04ca6eac86090` | `09c38e89f64c3a569b6a87682d95cb9cebe707623b6d601cb0d66ab635a219f6` | code digest `fdf25907cfe2404b86b46b62cb9b853831f9aa48b4e4c7bf9edcb8d95379288f` |
| `enrichment/target/alvo_eda.py` | `c692979d1f07f1a4c5aff1029130befe9c9c113dd95ecbc5759930df7fee6b37` | `64c2654f84a08262f6b111c6428b5f3efcf147875321d539ce1d08f542e7e80c` | code digest `b90cfc7439f688d5ac82905be9032fa2010862cee8da7591b2f9c2beff146f06` |
| `enrichment/target/alvo_enriquecimento.py` | `8bd7f85621da0b9e680e1c53152e9625fdac1aa4332a426d126d02990ca335d9` | `0ec6d723d99059f9236521bc23532020e90e9d2229cb17c292009555556d90f6` | code digest `4debcdf3abaad5a73070113ae446a72e309400fcc579e93eff194bf1e9d73aec` |
| `enrichment/target/analise_proximidade_erb.py` | `8b97854f670f74ecd3c8b109d7a4c8f1df7f2bb89725d3167d47e4ffd2377de0` | `e7a401b1faeb15d7f846d96485ac2cc805e6eaa0560a054ad7cbe37711708f40` | code digest `439a6071709d33407b75d03df58f252ad7eefe8dd6b5f4f9cbb0744746a56264` |
| `enrichment/target/run_alvo_pipeline.py` | `18bd0c72b1e33171b9c99470ce4755e5980c227a4590c7b40af600293cb74a62` | `8017a811281350e5163a3ba9cae7142f7badcae2b02f896d1b289145c9d12236` | code digest `f5fa4797c8454c89d760d1f0106494928d36b26fbe9549ff6a9e7429b36d8463` |
| `enrichment/triangulation/run_triangulation.py` | `3606d90ed14488c0c32ccef4dc75793c517bb0ed7d1d2e210fee5ca020eb6564` | `d23271c2d5f0a284256467b68cd90f94302c9d9c9bc24b2e65f4833a46779949` | code digest `16258e1d876f00f995c7334f67a0918133629ec4414ded10019f83cf000be39b` |
| `enrichment/triangulation/run_triangulation_optimized.py` | `a30dee6b61b1ef01fc6b50aab7a39e82c41959d0ad3e3f58572b3f27c1eb810b` | `ba020c80f16f681f01727c30bacfd66679b3f14850b317d4966bd8228d71453e` | code digest `ff4d70bd021903a40ae3730d257d0731753e9763d9cb5915b55d4579f1ded34b` |
| `enrichment/triangulation/triangulation_config.yaml` | `eb1bf62ea8b2512479fd98e9800d83f7c55f06009e0c48dbf91e18f2be300531` | `44cabc2d1a69d1c39a5dfff6b6685098a8bd349d9d0fc8a37dd5ad7e5adf95b0` | code digest `529a246eed5210565fc3984c957c4d741f0892cafff583e1db84b97795e4a819` |
| `enrichment/triangulation/triangulation_ctc_ctnc.py` | `7a0a0c318f762a4df1f9b140d3b15713bacb915e6c823bbda64f9628f4c24b1e` | `c4394d4c0031433b1b9ce582fbca35b4beba2563aca90dc9f62e461e0910fc2d` | code digest `5a172f6e70b27cfcd78ab3b91bc679d8f02c9fd41f7a891b6cd75902210a9746` |
| `environment.txt` | n/a (written for this package) | `ff9b391725c32cfd4403efe55fc83b6a5cbbaf26d1f6f600b8d64dbd50f0255a` | new file |
| `graph_construction/grafo_v3.py` | `3853ee0ff8850015497b6bd622152fafe79f33c97b967de99918709255ae24b4` | `db9fd65b0716de6e0c0165feb35613f44abfc0aa84ad56627f5ea0a6d4de618f` | code digest `1d46554ea0bc51b683f2466a9f22357edc9a05fb181ff6e43bd86da7d38280fb` |
| `graph_construction/grafo_v3.yaml` | `db8b1424652c9c54358f5edae2ba3f937104ef25c973cf8358ddd71622c43501` | `9e8aac3e296fecf0154c170d65d5e3af5bd5fc6174203bb6752ae5c2d1f87e4b` | code digest `bd28b97941740f509b58ab1997ad3125a06c8559ea6f5d3c46dc1b46df1e841e` |
| `graph_construction/run_grafo_v3.py` | `96333c9142f29201c2dd229921451c961db094fd5aa4c532375f5be288e16bf6` | `5ab5c103dc04486607fe92b358a5260a7af2ca36803d888e2e77e54a9345ff02` | code digest `79ccdafd0988c03be38e556c61c39565e0f1dd818ee6d4a84d9d5181f23360d8` |
| `normalization/eda_normalizacao_v3.py` | `a10eb213dc4cf24b1e715e2f07905c2e659a3a9e486a95ee394fdbc480516133` | `84369267ce0588a253412eb87dfee09705c97961dd131f01acf1b72214730a32` | code digest `da933209396bf77ca4d8b904d9a19013e9f32a16965b9ea0a7c23cd81874a18b` |
| `normalization/id_mapeamento_v3.py` | `c7678317a9a820647fe956946094fc3f963ec7a81b75e373dc850d3f7f9ae103` | `2706b3aa897e7193b4484ebd8823b1833681560e9157ed18b7816cfd48340ab6` | code digest `4fa76278f4aa1c7c026ca23b672a38639126b854099a6c239163abdbdede1072` |
| `normalization/normalizacao_v3.py` | `d5c7bf6adba8a96ecafbcc0c60bd1078213f89ea0295ee0accb64195d8c90f3d` | `23a345fcc2b1fbc73d220eb29eb7c78c9a1f9fe20b192442e6a2efbd11bc7abb` | code digest `e2f13374bc8d2e574fde68590417998aceddb1d7b9a82c3e32be1b600c00b9d3` |
| `normalization/normalizacao_v3.yaml` | `12847e5ae8b5a47df817a5fa06979133e90e3643ab0ced7bd8f52529fe436f90` | `79af2c9b2c02f518da8be2ce3a6b18130318df2a51099e732f7fe0f7ea2f0f3b` | code digest `c26eaf4a712dbd37024cc9b7f1d2c1eaf7ff17a24ea29d480b23f0c622c8fc76` |
| `normalization/pipeline_v3.py` | `1838e903d73a13467ee22260b283e5400b8b1fcb0d6fe69bc30b2611e5cd425e` | `22fec3d123d8e9bce600e41c223b11050a5f0aafa635c171d6511f7b8433204c` | code digest `f7961c460cbcb3849290859d1eac3034b27804f086de13d78092c9d5da61d4b1` |
| `normalization/run_normalizacao_v3.py` | `0b7776f54780506a36b0528d832bb3374be5d47b9f88a9fbefd282c88f547b38` | `ee0b3b622f4653d7ea3569a433fc28651fa717253485fee47d4107d3151f58d1` | code digest `552e770c2258bec0d8dcd6fcb32f38bd5f81e2047bba4b9a403694e257219533` |
| `training/ARPIA_Sensitivity_Parallel.py` | `d246c9291128849fb17d6a299b898d601ce84a4334657b7f0b2d079807ad3914` | `c366fe3f9f7e049255706379df6b2f1606e5debf9cbcc0d7fc876120de2544df` | code digest `334630aba7e52229d86e52760221abe32482622816bde63eb10cef2961137ffe` |
| `training/ablation_study_arpia_v3.py` | `74c5ec15a6ff66662f8cdf3d1a4e6a55210a47ac1e43e263f0f17bba0aca018e` | `1538304b1b66dbdc2d257549fffd2c2db051827cb7aeebf1cbd39b1672882eb1` | code digest `3cd398e2d08ee9da4e12cbef8ad9cf8f268f1c7595d79b90be9b4dad5842bb53` |
| `training/ablation_study_continuation.py` | `61c585350a57b94128f29d33429ee5d49290f9fb0954c51c501de362510cf84c` | `8c8e1b4b3186a4041bd8e77ff5106dc7e61f4ebf871303263356ddc1c6f38b75` | code digest `ae67c718b25966a95d905692a4ac9fe48a27e7c34ed66bacb80fe538c04fe441` |
| `training/ablation_study_continuation_2.py` | `bc41ab1091ff4c0371eb0a48a5d6c8441b2ecf58465dde55b50da0f18042c4b3` | `3673641d08e267bb3709090745aad7a2d7f7451582d35a84e9ab57e912ff5c25` | code digest `ef33b96bf8c70c379aa00888c03a873ca2b5973757c9580f035b2492f62ddc39` |
| `training/ablation_study_continuation_3.py` | `37beabe98350373662150dd2ff1500f82bbede5883fab6a483ad801dfbd6d8b4` | `74c3a3b76b2f27f211fa2072786781aa251a324c886389162b92f6951e7a00b1` | code digest `678504d75733c55455ac3a9b8003bd267b50d3dd458faa2aaea4ffd9d7fe7efe` |
| `training/arpia_gatv2.py` | `d81e99312a9e9aae9af846d96fbee4dd7d744e964cad3aa7efcd57add909a80b` | `acae3f74b4e16eae637ddb494740c931e16f5b0a60438b2ae3ad4af68421c8b6` | code digest `3fe17f9ccfb258c283a51423ae01ce4d22ea707b958882a9fd85df57c784f212` |
| `training/arpia_han.py` | `b1f4162d91a71616e02dce1b2435946fe81ff176de28c2c3dcae0b9a7054f3b0` | `fc0a30bc7eb1de5fed69b9e993b23027450e0fd409917fb95f1309180eb8f9a3` | code digest `adc29a7357c4a50c59c21b3c670773f730efbd1c7e5ef9e1299a895e5473633a` |
| `training/arpia_rgcn.py` | `fa33cd74b1a340d01f697b539020bbe570fb7a061882df4fff2c900483f9f890` | `c0e4cf9da5d6bde41dd2940e4ec1ab8ee3201bc5d83ce2219dffcf1d3498157f` | code digest `d7481cdaab45ce53c21594ff5762a374a591a0ed3da6f9a64059245cc3632d70` |
| `training/baseline_models.py` | `64fed4b9416f9ea144d73e41f43e23abf7c52998103ac1c27a52f7c171e945f2` | `51afd64a2cc237e7d73abcd2dd982d0b3ac747a238c80a32bd2a2261294dac10` | code digest `59764302a716fb3d03975766f25c4afea4c71f1f92122947cd8716e4e04b4315` |
| `training/config_colab.py` | `fe1612251fead3c04b8924b1da49b26ba98f5dfc54d155c05ff82572acb1dc9e` | `abb3da6ba35b7554459aeb547a2bada46104d2a93939ea80bd344452d25f23e9` | code digest `344e424eeeaeb8f4b1fe31cac9f7a14ca593596627586ef4448137134865b59c` |
| `training/data_loaders.py` | `1af0abc2b276d56285c65949ed44b69319f6de298ec41768595f208b8bae2ffe` | `2e054f52cd798e8a65ee6b023e865b7d41f51dbca1752bd2caa98ef4215c51ad` | code digest `3b3f6853441ef504afdb1b96741856f842aa5e49711407ccd121b843e3834c3b` |
| `training/losses.py` | `3c22f7ab86a34708fa42b820e22c434d477885d11a2d926457d6a6054b787ef6` | `a6bd7266cad71c197bec1d053de394fd51e78c29b0c32adf35225ef69af5459a` | code digest `55198c31c6d246c369ede0fc33e9af3624147e973a783eeb2a6d97716629fa98` |
| `training/models.py` | `750c1edec11d19f0e475fe5a9b9a554c9a4de46ba5e45ded7a21d511551be509` | `c6ce739974cd7451b0e6885c0dfb51e58be6a893d17985d7c2c5a76d951513fa` | code digest `fd67cf527806eabbb95f68cfe78f4529f45aafc505f630c0dcd9dfd71e30313a` |
| `training/run_ablation_arpia.py` | `ff0c0380e6d1438506dc4f10cdceccec097ee6d975d084eb2cefd9076af1e8b9` | `4446412c8104f5801285490a97d6bca31ea7118a676983abb70fee978dfda5c2` | code digest `69a901c8422ec82c92b6af0fcdf9ff22332ccdb27912646c147d6689368731c0` |
| `training/run_continuation.py` | `bd4d14493c73ca93636bf3a1f76ef7a0f88242b8546934e6df97ff2ef95b3597` | `2d5d88aa488a1e9c59a5cdfc093f4d2fb7f26e72026e5f4c93683cc0dfab1cef` | code digest `bac450cd97726c701ad34d08a0a6e0ebb262bc004a6648ddc8e1d481a37c768c` |
| `training/train_colab.py` | `a50d8bf12933eff98e1f1cc7a56f88d402e111ddb5e782ca0d59a5680079f062` | `6cff3f50029d028ba907bff1cd0fd55875c6db1182f981e9facbcc34c139e572` | code digest `302eb60b4eadab2ccd24ea59c7d824e5badc132a8163bf6445240c56b396511b` |
| `training/trainers.py` | `78d8aafac2981f37fb03bfe7d77ad3492dcdf42721dbbf4799dd964da5863d89` | `853dd84948ea4a3a868aeba0ddebb20014e8e392f25cde301101b76664fb7a46` | code digest `4076d0df5a78be520237b498e75ea014331b63f82271dee90a9a1fe616e81057` |
| `training/utils.py` | `2e013244ee71fca4ccaa590c549291cad7ef58372412c090bb2263547629351a` | `6496e2affbe0412cbf66cd00639fe3b8d125cd8ce8108df8f5f83bf7dc346c07` | code digest `adfa8709a023dd1cf298819dbeb6f6c986a1f51fdc325e825907cb2c0862dffc` |
| `verify_code.py` | n/a (written for this package) | `14f9ba2a2acf5fa310d5f75b38fb7c6ce3c93202b3de90584f8e815414089504` | new file |
