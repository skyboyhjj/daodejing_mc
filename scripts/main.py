# -*- coding: utf-8 -*-
"""
《道德经》马尔科夫链建模 - 主流程
基于文档2（daodejing_full_text-带章节.txt 的 Python 字典版本）
含第10章校订 + 扩充概念词典 + k=1/k=2 转移矩阵 + 可视化

【重构说明 · 2026-08-21 T12】
  - 环境配置（UTF-8 / 中文字体 / 路径）抽到 core.env
  - 核心算法（清洗 / 转移矩阵 / EI / 粗粒化）抽到 core.pipeline
  - main.py 保留对外 API 与可视化逻辑，作为全链路唯一入口
"""

import numpy as np
import pandas as pd
import re
import json
import os
import sys

# 确保能 import core 包（core/ 位于项目根目录 = 本脚本的上一级）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score

from core.env import (
    setup_env, get_cn_font, CN_FONT, BASE_DIR, OUTPUT_DIR,
    setup_utf8_stdio,
)
from core.pipeline import (
    clean_text, extract_concepts, build_full_sequence,
    build_transition_matrix, stationary_distribution,
    effective_information, normalized_ei, lumpability_error,
    svd_coarse_grain, build_macro_transition,
    semantic_macro_labels as _semantic_macro_labels_impl,
    extract_concepts as _extract_concepts_impl,
    build_full_sequence as _build_full_sequence_impl,
    svd_embedding,
)

# 重新导出核心函数，保持 `from main import ...` 的向后兼容
from core.pipeline import (
    clean_text, extract_concepts, build_full_sequence,
    build_transition_matrix, stationary_distribution,
    effective_information, normalized_ei, lumpability_error,
    svd_coarse_grain, build_macro_transition,
)

# 统一环境配置（UTF-8 + 中文字体）
setup_env()

# ============================================================
# 文档2 数据（81章完整文本，已校订第10章）
# ============================================================
DAODEJING = {
    1: "道可道，非常道。名可名，非常名。无名天地之始。有名万物之母。故常无欲以观其妙，常有欲以观其徼。此两者同出而异名，同谓之玄。玄之又玄，众妙之门。",
    2: "天下皆知美之为美，斯恶已。皆知善之为善，斯不善已。故有无相生，难易相成，长短相形，高下相倾，音声相和，前后相随。是以圣人处无为之事，行不言之教。万物作焉而不辞，生而不有，为而不恃，功成而弗居。夫唯弗居，是以不去。",
    3: "不尚贤，使民不争。不贵难得之货，使民不为盗。不见可欲，使民心不乱。是以圣人之治，虚其心，实其腹，弱其志，强其骨。常使民无知无欲，使夫智者不敢为也。为无为，则无不治。",
    4: "道冲而用之或不盈，渊兮似万物之宗。挫其锐，解其纷，和其光，同其尘。湛兮似或存，吾不知谁之子，象帝之先。",
    5: "天地不仁，以万物为刍狗；圣人不仁，以百姓为刍狗。天地之间，其犹橐籥乎？虚而不屈，动而愈出。多言数穷，不如守中。",
    6: "谷神不死，是谓玄牝。玄牝之门，是谓天地根。绵绵若存，用之不勤。",
    7: "天长地久。天地所以能长且久者，以其不自生，故能长生。是以圣人后其身而身先，外其身而身存。非以其无私邪？故能成其私。",
    8: "上善若水。水善利万物而不争，处众人之所恶，故几于道。居善地，心善渊，与善仁，言善信，正善治，事善能，动善时。夫唯不争，故无尤。",
    9: "持而盈之，不如其已。揣而锐之，不可长保。金玉满堂，莫之能守。富贵而骄，自遗其咎。功遂身退，天之道也。",
    # 第10章已校订：爱国治民（王弼定本）而非 爱民治国
    10: "载营魄抱一，能无离乎？专气致柔，能婴儿乎？涤除玄览，能无疵乎？爱国治民，能无知乎？天门开阖，能为雌乎？明白四达，能无为乎？生之畜之，生而不有，为而不恃，长而不宰，是谓玄德。",
    11: "三十辐共一毂，当其无，有车之用。埏埴以为器，当其无，有器之用。凿户牖以为室，当其无，有室之用。故有之以为利，无之以为用。",
    12: "五色令人目盲，五音令人耳聋，五味令人口爽，驰骋畋猎令人心发狂，难得之货令人行妨。是以圣人为腹不为目，故去彼取此。",
    13: "宠辱若惊，贵大患若身。何谓宠辱若惊？宠为下，得之若惊，失之若惊，是谓宠辱若惊。何谓贵大患若身？吾所以有大患者，为吾有身，及吾无身，吾有何患？故贵以身为天下，若可寄天下；爱以身为天下，若可托天下。",
    14: "视之不见名曰夷，听之不闻名曰希，搏之不得名曰微。此三者不可致诘，故混而为一。其上不皦，其下不昧，绳绳不可名，复归于无物。是谓无状之状，无物之象，是谓惚恍。迎之不见其首，随之不见其后。执古之道，以御今之有。能知古始，是谓道纪。",
    15: "古之善为道者，微妙玄通，深不可识。夫唯不可识，故强为之容：豫兮若冬涉川，犹兮若畏四邻，俨兮其若客，涣兮若冰之将释，敦兮其若朴，旷兮其若谷，混兮其若浊。孰能浊以静之徐清？孰能安以动之徐生？保此道者不欲盈。夫唯不盈，故能蔽而新成。",
    16: "致虚极，守静笃。万物并作，吾以观复。夫物芸芸，各复归其根。归根曰静，静曰复命。复命曰常，知常曰明。不知常，妄作凶。知常容，容乃公，公乃王，王乃天，天乃道，道乃久，没身不殆。",
    17: "太上，不知有之；其次，亲而誉之；其次，畏之；其次，侮之。信不足焉，有不信焉。悠兮其贵言。功成事遂，百姓皆谓我自然。",
    18: "大道废，有仁义；智慧出，有大伪；六亲不和，有孝慈；国家昏乱，有忠臣。",
    19: "绝圣弃智，民利百倍；绝仁弃义，民复孝慈；绝巧弃利，盗贼无有。此三者以为文不足，故令有所属：见素抱朴，少私寡欲。",
    20: "绝学无忧。唯之与阿，相去几何？美之与恶，相去若何？人之所畏，不可不畏。荒兮其未央哉！众人熙熙，如享太牢，如春登台。我独泊兮其未兆，如婴儿之未孩。傫傫兮若无所归。众人皆有余，而我独若遗。我愚人之心也哉！沌沌兮！俗人昭昭，我独昏昏。俗人察察，我独闷闷。澹兮其若海，飂兮若无止。众人皆有以，而我独顽似鄙。我独异于人，而贵食母。",
    21: "孔德之容，惟道是从。道之为物，惟恍惟惚。惚兮恍兮，其中有象；恍兮惚兮，其中有物。窈兮冥兮，其中有精；其精甚真，其中有信。自今及古，其名不去，以阅众甫。吾何以知众甫之状哉？以此。",
    22: "曲则全，枉则直，洼则盈，敝则新，少则得，多则惑。是以圣人抱一为天下式。不自见故明，不自是故彰，不自伐故有功，不自矜故长。夫唯不争，故天下莫能与之争。古之所谓曲则全者，岂虚言哉！诚全而归之。",
    23: "希言自然。故飘风不终朝，骤雨不终日。孰为此者？天地。天地尚不能久，而况于人乎？故从事于道者，同于道；德者，同于德；失者，同于失。同于道者，道亦乐得之；同于德者，德亦乐得之；同于失者，失亦乐得之。信不足焉，有不信焉。",
    24: "企者不立，跨者不行，自见者不明，自是者不彰，自伐者无功，自矜者不长。其在道也，曰余食赘行。物或恶之，故有道者不处。",
    25: "有物混成，先天地生。寂兮寥兮，独立不改，周行而不殆，可以为天下母。吾不知其名，字之曰道，强为之名曰大。大曰逝，逝曰远，远曰反。故道大，天大，地大，王亦大。域中有四大，而王居其一焉。人法地，地法天，天法道，道法自然。",
    26: "重为轻根，静为躁君。是以圣人终日行不离辎重。虽有荣观，燕处超然。奈何万乘之主，而以身轻天下？轻则失根，躁则失君。",
    27: "善行无辙迹，善言无瑕谪，善数不用筹策，善闭无关楗而不可开，善结无绳约而不可解。是以圣人常善救人，故无弃人；常善救物，故无弃物，是谓袭明。故善人者，不善人之师；不善人者，善人之资。不贵其师，不爱其资，虽智大迷，是谓要妙。",
    28: "知其雄，守其雌，为天下溪。为天下溪，常德不离，复归于婴儿。知其白，守其黑，为天下式。为天下式，常德不忒，复归于无极。知其荣，守其辱，为天下谷。为天下谷，常德乃足，复归于朴。朴散则为器，圣人用之则为官长。故大制不割。",
    29: "将欲取天下而为之，吾见其不得已。天下神器，不可为也。为者败之，执者失之。故物或行或随，或歔或吹，或强或羸，或挫或隳。是以圣人去甚，去奢，去泰。",
    30: "以道佐人主者，不以兵强天下，其事好还。师之所处，荆棘生焉。大军之后，必有凶年。善有果而已，不敢以取强。果而勿矜，果而勿伐，果而勿骄，果而不得已，果而勿强。物壮则老，是谓不道，不道早已。",
    31: "夫兵者，不祥之器。物或恶之，故有道者不处。君子居则贵左，用兵则贵右。兵者不祥之器，非君子之器。不得已而用之，恬淡为上，胜而不美。而美之者，是乐杀人。夫乐杀人者，则不可得志于天下矣。吉事尚左，凶事尚右。偏将军居左，上将军居右，言以丧礼处之。杀人之众，以悲哀泣之，战胜以丧礼处之。",
    32: "道常无名，朴虽小，天下莫能臣也。侯王若能守之，万物将自宾。天地相合以降甘露，民莫之令而自均。始制有名，名亦既有，夫亦将知止。知止可以不殆。譬道之在天下，犹川谷之于江海。",
    33: "知人者智，自知者明。胜人者有力，自胜者强。知足者富，强行者有志。不失其所者久，死而不亡者寿。",
    34: "大道泛兮，其可左右。万物恃之而生而不辞，功成不名有。衣养万物而不为主，常无欲，可名于小；万物归焉而不为主，可名为大。以其终不自为大，故能成其大。",
    35: "执大象，天下往。往而不害，安平太。乐与饵，过客止。道之出口，淡乎其无味，视之不足见，听之不足闻，用之不足既。",
    36: "将欲歙之，必固张之；将欲弱之，必固强之；将欲废之，必固兴之；将欲夺之，必固与之。是谓微明。柔弱胜刚强。鱼不可脱于渊，国之利器不可以示人。",
    37: "道常无为而无不为。侯王若能守之，万物将自化。化而欲作，吾将镇之以无名之朴。无名之朴，夫亦将无欲。不欲以静，天下将自定。",
    38: "上德不德，是以有德；下德不失德，是以无德。上德无为而无以为，下德为之而有以为。上仁为之而无以为，上义为之而有以为。上礼为之而莫之应，则攘臂而扔之。故失道而后德，失德而后仁，失仁而后义，失义而后礼。夫礼者，忠信之薄而乱之首。前识者，道之华而愚之始。是以大丈夫处其厚，不居其薄；处其实，不居其华。故去彼取此。",
    39: "昔之得一者：天得一以清，地得一以宁，神得一以灵，谷得一以盈，万物得一以生，侯王得一以为天下贞。其致之。天无以清将恐裂，地无以宁将恐发，神无以灵将恐歇，谷无以盈将恐竭，万物无以生将恐灭，侯王无以贵高将恐蹶。故贵以贱为本，高以下为基。是以侯王自谓孤、寡、不谷。此非以贱为本邪？非乎？故致数舆无舆。不欲琭琭如玉，珞珞如石。",
    40: "反者道之动，弱者道之用。天下万物生于有，有生于无。",
    41: "上士闻道，勤而行之；中士闻道，若存若亡；下士闻道，大笑之。不笑不足以为道。故建言有之：明道若昧，进道若退，夷道若颣。上德若谷，大白若辱，广德若不足，建德若偷，质真若渝。大方无隅，大器晚成，大音希声，大象无形。道隐无名。夫唯道，善贷且成。",
    42: "道生一，一生二，二生三，三生万物。万物负阴而抱阳，冲气以为和。人之所恶，唯孤、寡、不谷，而王公以为称。故物或损之而益，或益之而损。人之所教，我亦教之。强梁者不得其死，吾将以为教父。",
    43: "天下之至柔，驰骋天下之至坚。无有入无间，吾是以知无为之有益。不言之教，无为之益，天下希及之。",
    44: "名与身孰亲？身与货孰多？得与亡孰病？是故甚爱必大费，多藏必厚亡。知足不辱，知止不殆，可以长久。",
    45: "大成若缺，其用不弊。大盈若冲，其用不穷。大直若屈，大巧若拙，大辩若讷。躁胜寒，静胜热。清静为天下正。",
    46: "天下有道，却走马以粪；天下无道，戎马生于郊。祸莫大于不知足，咎莫大于欲得。故知足之足，常足矣。",
    47: "不出户，知天下；不窥牖，见天道。其出弥远，其知弥少。是以圣人不行而知，不见而名，不为而成。",
    48: "为学日益，为道日损。损之又损，以至于无为。无为而无不为。取天下常以无事，及其有事，不足以取天下。",
    49: "圣人无常心，以百姓心为心。善者吾善之，不善者吾亦善之，德善。信者吾信之，不信者吾亦信之，德信。圣人在天下歙歙，为天下浑其心。百姓皆注其耳目，圣人皆孩之。",
    50: "出生入死。生之徒十有三，死之徒十有三。人之生，动之死地亦十有三。夫何故？以其生生之厚。盖闻善摄生者，陆行不遇兕虎，入军不被甲兵。兕无所投其角，虎无所措其爪，兵无所容其刃。夫何故？以其无死地。",
    51: "道生之，德畜之，物形之，势成之。是以万物莫不尊道而贵德。道之尊，德之贵，夫莫之命而常自然。故道生之，德畜之，长之育之，亭之毒之，养之覆之。生而不有，为而不恃，长而不宰，是谓玄德。",
    52: "天下有始，以为天下母。既得其母，以知其子；既知其子，复守其母，没身不殆。塞其兑，闭其门，终身不勤。开其兑，济其事，终身不救。见小曰明，守柔曰强。用其光，复归其明，无遗身殃，是为习常。",
    53: "使我介然有知，行于大道，唯施是畏。大道甚夷，而民好径。朝甚除，田甚芜，仓甚虚。服文彩，带利剑，厌饮食，财货有余，是为盗夸。非道也哉！",
    54: "善建者不拔，善抱者不脱，子孙以祭祀不辍。修之于身，其德乃真；修之于家，其德乃余；修之于乡，其德乃长；修之于国，其德乃丰；修之于天下，其德乃普。故以身观身，以家观家，以乡观乡，以国观国，以天下观天下。吾何以知天下然哉？以此。",
    55: "含德之厚，比于赤子。毒虫不螫，猛兽不据，攫鸟不搏。骨弱筋柔而握固。未知牝牡之合而全作，精之至也。终日号而不嗄，和之至也。知和曰常，知常曰明，益生曰祥，心使气曰强。物壮则老，谓之不道，不道早已。",
    56: "知者不言，言者不知。塞其兑，闭其门，挫其锐，解其纷，和其光，同其尘，是谓玄同。故不可得而亲，不可得而疏；不可得而利，不可得而害；不可得而贵，不可得而贱。故为天下贵。",
    57: "以正治国，以奇用兵，以无事取天下。吾何以知其然哉？以此。天下多忌讳，而民弥贫；民多利器，国家滋昏；人多伎巧，奇物滋起；法令滋彰，盗贼多有。故圣人云：我无为而民自化，我好静而民自正，我无事而民自富，我无欲而民自朴。",
    58: "其政闷闷，其民淳淳；其政察察，其民缺缺。祸兮福之所倚，福兮祸之所伏。孰知其极？其无正。正复为奇，善复为妖。人之迷，其日固久。是以圣人方而不割，廉而不刿，直而不肆，光而不耀。",
    59: "治人事天，莫若啬。夫唯啬，是谓早服。早服谓之重积德，重积德则无不克，无不克则莫知其极，莫知其极可以有国，有国之母可以长久。是谓深根固柢，长生久视之道。",
    60: "治大国若烹小鲜。以道莅天下，其鬼不神。非其鬼不神，其神不伤人。非其神不伤人，圣人亦不伤人。夫两不相伤，故德交归焉。",
    61: "大邦者下流，天下之交，天下之牝。牝常以静胜牡，以静为下。故大邦以下小邦，则取小邦；小邦以下大邦，则取大邦。故或下以取，或下而取。大邦不过欲兼畜人，小邦不过欲入事人。夫两者各得所欲，大者宜为下。",
    62: "道者万物之奥，善人之宝，不善人之所保。美言可以市尊，美行可以加人。人之不善，何弃之有？故立天子，置三公，虽有拱璧以先驷马，不如坐进此道。古之所以贵此道者何？不曰以求得，有罪以免邪？故为天下贵。",
    63: "为无为，事无事，味无味。大小多少，报怨以德。图难于其易，为大于其细。天下难事必作于易，天下大事必作于细。是以圣人终不为大，故能成其大。夫轻诺必寡信，多易必多难。是以圣人犹难之，故终无难矣。",
    64: "其安易持，其未兆易谋，其脆易泮，其微易散。为之于未有，治之于未乱。合抱之木，生于毫末；九层之台，起于累土；千里之行，始于足下。为者败之，执者失之。是以圣人无为故无败，无执故无失。民之从事，常于几成而败之。慎终如始，则无败事。是以圣人欲不欲，不贵难得之货；学不学，复众人之所过。以辅万物之自然，而不敢为。",
    65: "古之善为道者，非以明民，将以愚之。民之难治，以其智多。故以智治国，国之贼；不以智治国，国之福。知此两者亦稽式。常知稽式，是谓玄德。玄德深矣，远矣，与物反矣，然后乃至大顺。",
    66: "江海所以能为百谷王者，以其善下之，故能为百谷王。是以欲上民，必以言下之；欲先民，必以身后之。是以圣人处上而民不重，处前而民不害。是以天下乐推而不厌。以其不争，故天下莫能与之争。",
    67: "天下皆谓我道大，似不肖。夫唯大，故似不肖。若肖，久矣其细也夫！我有三宝，持而保之：一曰慈，二曰俭，三曰不敢为天下先。慈故能勇，俭故能广，不敢为天下先，故能成器长。今舍慈且勇，舍俭且广，舍后且先，死矣！夫慈，以战则胜，以守则固。天将救之，以慈卫之。",
    68: "善为士者不武，善战者不怒，善胜敌者不与，善用人者为之下。是谓不争之德，是谓用人之力，是谓配天古之极。",
    69: "用兵有言：吾不敢为主而为客，不敢进寸而退尺。是谓行无行，攘无臂，扔无敌，执无兵。祸莫大于轻敌，轻敌几丧吾宝。故抗兵相加，哀者胜矣。",
    70: "吾言甚易知，甚易行。天下莫能知，莫能行。言有宗，事有君。夫唯无知，是以不我知。知我者希，则我者贵。是以圣人被褐怀玉。",
    71: "知不知，上；不知知，病。夫唯病病，是以不病。圣人不病，以其病病，是以不病。",
    72: "民不畏威，则大威至。无狎其所居，无厌其所生。夫唯不厌，是以不厌。是以圣人自知不自见，自爱不自贵。故去彼取此。",
    73: "勇于敢则杀，勇于不敢则活。此两者或利或害。天之所恶，孰知其故？是以圣人犹难之。天之道，不争而善胜，不言而善应，不召而自来，繟然而善谋。天网恢恢，疏而不失。",
    74: "民不畏死，奈何以死惧之？若使民常畏死，而为奇者，吾得执而杀之，孰敢？常有司杀者杀。夫代司杀者杀，是谓代大匠斫。夫代大匠斫者，希有不伤其手矣。",
    75: "民之饥，以其上食税之多，是以饥。民之难治，以其上之有为，是以难治。民之轻死，以其上求生之厚，是以轻死。夫唯无以生为者，是贤于贵生。",
    76: "人之生也柔弱，其死也坚强。万物草木之生也柔脆，其死也枯槁。故坚强者死之徒，柔弱者生之徒。是以兵强则灭，木强则折。强大处下，柔弱处上。",
    77: "天之道，其犹张弓与？高者抑之，下者举之；有余者损之，不足者补之。天之道，损有余而补不足。人之道则不然，损不足以奉有余。孰能有余以奉天下？唯有道者。是以圣人为而不恃，功成而不处，其不欲见贤。",
    78: "天下莫柔弱于水，而攻坚强者莫之能胜，以其无以易之。弱之胜强，柔之胜刚，天下莫不知，莫能行。是以圣人云：受国之垢，是谓社稷主；受国不祥，是为天下王。正言若反。",
    79: "和大怨，必有余怨，安可以为善？是以圣人执左契，而不责于人。有德司契，无德司彻。天道无亲，常与善人。",
    80: "小国寡民，使有什伯之器而不用，使民重死而不远徙。虽有舟舆，无所乘之；虽有甲兵，无所陈之。使人复结绳而用之。甘其食，美其服，安其居，乐其俗。邻国相望，鸡犬之声相闻，民至老死不相往来。",
    81: "信言不美，美言不信。善者不辩，辩者不善。知者不博，博者不知。圣人不积，既以为人己愈有，既以与人己愈多。天之道，利而不害。圣人之道，为而不争。"
}

# ============================================================
# 扩充概念词典（覆盖81章核心哲学范畴）
# ============================================================
CONCEPT_DICT = {
    # ---- 道体论 ----
    "道": ["道", "大道", "有道", "无道", "闻道", "为道", "道纪", "道法"],
    "德": ["德", "玄德", "上德", "下德", "不德", "有德", "无德", "稽式"],
    "自然": ["自然"],
    "无为": ["无为", "不争", "不为", "不敢为", "不敢", "不得已", "无事"],
    "无": ["无", "无名", "无欲", "无所", "无以", "无为", "无有", "无间", "无尤", "无死地", "无兵"],
    "有": ["有", "有名", "有为", "有之", "有天下", "有身", "有知"],
    "一": ["一", "抱一", "得一", "为一"],
    "朴": ["朴", "素", "无名之朴", "见素抱朴"],
    # ---- 辩证法 ----
    "反": ["反", "反者", "复", "归根", "观复", "复命"],
    "柔弱": ["柔弱", "柔", "弱者", "弱", "至柔", "柔脆"],
    "刚强": ["刚强", "坚强", "强", "壮", "盈", "满"],
    "辩证": ["有无相生", "难易相成", "长短相形", "高下相倾", "音声相和", "前后相随",
             "祸福", "祸兮福", "福兮祸", "正言若反", "曲则全", "枉则直", "洼则盈",
             "敝则新", "少则得", "多则惑"],
    # ---- 圣人/治术 ----
    "圣人": ["圣人", "大丈夫", "善为道者", "善为士者", "善建者", "善抱者", "善摄生者"],
    "治": ["治", "治国", "不治", "自化", "自正", "自富", "自朴", "以正治国"],
    "民": ["民", "百姓", "众人", "俗人", "天下"],
    "侯王": ["侯王", "王", "万乘之主", "天子", "三公"],
    # ---- 兵/战争 ----
    "兵": ["兵", "用兵", "大军", "戎马", "走马", "战胜", "偏将军", "上将军"],
    # ---- 修身处世 ----
    "知足": ["知足", "知止", "不辱", "不殆", "长久", "长生"],
    "慈": ["慈", "俭", "三宝"],
    "知": ["知", "自知", "知人", "知常", "明", "聪明"],
    "欲": ["欲", "可欲", "不欲", "贵欲", "欲得", "贪"],
    # ---- 宇宙论 ----
    "天地": ["天地", "天", "地", "天下母", "万物之始", "混成"],
    "象": ["象", "大象", "无状之状", "惚恍", "恍惚", "微明"],
    "玄": ["玄", "玄牝", "玄览", "玄同", "玄妙"],
    # ---- 方法论 ----
    "不言": ["不言", "无言", "希言", "贵言", "多言"],
    "守": ["守", "守中", "守静", "守柔", "守其母", "守辱"],
    "去": ["去", "去甚", "去奢", "去泰", "去彼取此"],
    "三宝": ["三宝", "慈", "俭", "不敢为天下先"],
    # ---- 特殊概念 ----
    "水": ["水", "上善若水"],
    "小国寡民": ["小国寡民", "小邦", "大邦"],
    "信": ["信", "不信", "美言不信", "信言不美"],
    "名": ["名", "可名", "无名", "有名", "名亦既有"],
}

# 构建反向映射（变体 → 标准化概念）
# 注意：同一变体可能被多个标准概念收录（如 "无为" 同时属于 "无为" 与 "无" 的变体）。
# 冲突时优先选择"更具体"（更长）的标准概念，避免 "无为" 被错误归入 "无"。
# 若长度相同（如 "不争"），按 CONCEPT_DICT 中标准概念的插入顺序优先（更先定义的覆盖）。
REVERSE_MAP = {}
for std, variants in CONCEPT_DICT.items():
    for v in variants:
        cur = REVERSE_MAP.get(v)
        if cur is None or len(std) > len(cur):
            REVERSE_MAP[v] = std

# 特殊多字概念优先匹配（按长度降序）
SPECIAL_MULTI = sorted(
    [v for v in REVERSE_MAP if len(v) >= 2],
    key=len, reverse=True
)
SINGLE_CHARS = ["道", "德", "无", "有", "一", "朴", "反", "柔", "弱",
                "强", "刚", "欲", "信", "名", "天", "地", "水", "兵",
                "民", "知", "治", "玄", "象", "守", "去", "慈", "俭"]


# ============================================================
# 宏观态分组（手工语义，M=6）—— 项目最终采用的分组方案
# 由 coarse_grain_v2.py 的多方案对比中胜出（可解释性优先）
# ============================================================
SEMANTIC_PARTITION = [
    ["道", "德", "玄", "象", "自然", "朴", "无", "有"],   # 0 道体论
    ["无为", "不言", "守", "去", "柔弱", "水"],             # 1 无为法
    ["反", "辩证", "刚强", "一"],                          # 2 辩证法
    ["圣人", "侯王", "治", "小国寡民", "兵"],                # 3 治术
    ["民", "欲", "知足", "知", "信", "名"],                 # 4 民知欲
    ["天地", "三宝"],                                      # 5 宇宙
]
MACRO_NAMES = ["道体论", "无为法", "辩证法", "治术", "民知欲", "宇宙"]


# ============================================================
# 备选分组：借鉴网页主题框架（hui-skill.org《道德经》8 大板块）
# 由"义理类别"导向改为"认知层级"导向（由体达用）：
#   道体论(认识道) → 辩证法(道的规律) → 修身内化(内化功夫)
#     → 无为论(方法论) → 治国论(外王之道) → 三宝(德之落实)
# 实测（tests/_experiment_web_framework.py）：
#   成块性误差 ε 0.004621→0.003834（-17%），宏观 EI 0.0119→0.0124，更优。
# 保留默认方案以保护既有结论；此为可切换的备选。
# ============================================================
SEMANTIC_PARTITION_WEB = [
    ["道", "德", "玄", "象", "自然", "朴", "无", "有", "天地"],  # 0 道体论（认识道）
    ["反", "辩证", "刚强", "柔弱", "一"],                       # 1 辩证法（道的规律）
    ["知足", "知", "欲", "信", "名", "守", "去", "不言"],        # 2 修身内化（内化功夫）
    ["无为", "水"],                                            # 3 无为论（方法论）
    ["圣人", "侯王", "治", "小国寡民", "民", "兵"],               # 4 治国论（外王之道）
    ["三宝"],                                                 # 5 三宝（德之落实）
]
MACRO_NAMES_WEB = ["道体论", "辩证法", "修身内化", "无为论", "治国论", "三宝"]


def semantic_macro_labels(idx, inv_idx):
    """
    根据手工语义分组为每个微观概念分配宏观标签（N 维向量）。
    供 build_outputs / export_visualization_data 等脚本复用。
    【重构 T12】实现抽到 core.pipeline.semantic_macro_labels（按参数传入分组）。
    """
    return _semantic_macro_labels_impl(idx, inv_idx, SEMANTIC_PARTITION)


# ============================================================
# 文本清洗与概念抽取
# 【重构 T12】clean_text 直接使用 core.pipeline 导入版本；
# extract_concepts / build_full_sequence 因依赖全局 REVERSE_MAP，
# 保留旧签名，内部委托给 core.pipeline 实现（按参数传入 reverse_map）。
# ============================================================

# 注意：clean_text 已从 core.pipeline 导入，此处不再重定义，避免覆盖。

def extract_concepts(text, chapter_num=None):
    """
    最长优先匹配 + 单字回退。
    返回该章的概念序列列表。
    【重构 T12】实现抽到 core.pipeline.extract_concepts（按参数传 REVERSE_MAP）。
    """
    return _extract_concepts_impl(text, REVERSE_MAP, chapter_num)


def build_full_sequence(chapter_texts, order='linear'):
    """
    构建全书概念序列。
    order='linear'：按1-81章顺序拼接
    order='sentence'：按句号切分，句内概念展开
    【重构 T12】实现抽到 core.pipeline.build_full_sequence（按参数传 REVERSE_MAP）。
    """
    return _build_full_sequence_impl(chapter_texts, REVERSE_MAP, order)


# ============================================================
# 核心算法（转移矩阵 / 平稳分布 / EI / 粗粒化）
# 【重构 T12】以下函数已全部抽到 core.pipeline 模块：
#   build_transition_matrix, stationary_distribution,
#   effective_information, normalized_ei, lumpability_error,
#   svd_coarse_grain, build_macro_transition
# main.py 在文件顶部从 core.pipeline 导入并重新导出，
# 保持 `from main import ...` 的向后兼容，此处不再重复定义。
# ============================================================


# ============================================================
# 可视化
# ============================================================
def plot_heatmap(P, idx, title, filename, figsize=(14, 12)):
    """转移矩阵聚类热力图"""
    labels = list(idx.keys())
    fig, ax = plt.subplots(figsize=figsize)
    sns.clustermap(P, annot=False, fmt=".2f", cmap="YlOrRd",
                   xticklabels=labels, yticklabels=labels,
                   figsize=figsize, dendrogram_ratio=0.15,
                   cbar_pos=(0.02, 0.8, 0.03, 0.18))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    # 也保存一个不带聚类的版本
    fig2, ax2 = plt.subplots(figsize=(12, 10))
    sns.heatmap(P, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=labels, yticklabels=labels, ax=ax2,
                cbar_kws={'label': '转移概率'})
    plt.title(title, fontsize=14, fontweight='bold')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename.replace('.png', '_annotated.png')),
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_steady_state(pi, idx, filename='steady_state.png'):
    """平稳分布条形图"""
    labels = list(idx.keys())
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = plt.cm.Spectral(np.linspace(0.2, 0.8, len(labels)))
    bars = ax.bar(range(len(labels)), pi, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=10)
    ax.set_ylabel('平稳概率 π', fontsize=12)
    ax.set_title('概念平稳分布（稳态下的概念权重）', fontsize=14, fontweight='bold')
    # 标注前5大
    top5 = np.argsort(pi)[-5:][::-1]
    for i in top5:
        ax.text(i, pi[i] + 0.002, f'{pi[i]:.3f}', ha='center', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()


def plot_spectral_scatter(embedding, labels, idx, inv_idx, s_values,
                          filename='spectral_scatter.png'):
    """SVD 谱空间散点图（2D + 3D）"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 2D 投影
    ax1 = axes[0]
    scatter = ax1.scatter(embedding[:, 0], embedding[:, 1],
                          c=labels, cmap='tab10', s=120, alpha=0.85, edgecolors='black', linewidth=0.5)
    for i in range(len(inv_idx)):
        ax1.annotate(inv_idx[i], (embedding[i, 0], embedding[i, 1]),
                      fontsize=8, ha='center', va='bottom', alpha=0.8)
    ax1.set_xlabel(f'SVD-1 (σ={s_values[0]:.4f})', fontsize=11)
    ax1.set_ylabel(f'SVD-2 (σ={s_values[1]:.4f})', fontsize=11)
    ax1.set_title('谱空间 2D 投影（K-Means 着色）', fontsize=13, fontweight='bold')
    ax1.axhline(0, color='gray', lw=0.3)
    ax1.axvline(0, color='gray', lw=0.3)
    ax1.grid(True, alpha=0.2)
    
    # 3D 投影
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                c=labels, cmap='tab10', s=100, alpha=0.85, edgecolors='black', linewidth=0.3)
    for i in range(len(inv_idx)):
        ax2.text(embedding[i, 0], embedding[i, 1], embedding[i, 2],
                 inv_idx[i], fontsize=7, ha='center')
    ax2.set_xlabel(f'SVD-1', fontsize=10)
    ax2.set_ylabel(f'SVD-2', fontsize=10)
    ax2.set_zlabel(f'SVD-3', fontsize=10)
    ax2.set_title('谱空间 3D 投影', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()


# 全局：检测 plotly + kaleido 是否真的可用
_PLOTLY_OK = False
try:
    from plotly.graph_objects import Sankey, Figure  # noqa
    import plotly.io as pio  # noqa
    # 验证 kaleido 能否真正工作（需要 Chrome）
    import kaleido  # noqa
    _PLOTLY_OK = True
except Exception:
    _PLOTLY_OK = False

def plot_sankey(P_macro, labels, macro_names, idx, inv_idx, Phi,
                filename='sankey.png'):
    """桑基图：微观→宏观 投影 + 宏观转移"""
    if not _PLOTLY_OK:
        print("  [plotly/kaleido 不可用] 使用 matplotlib 版桑基图")
        plot_sankey_matplotlib(P_macro, labels, macro_names, idx, inv_idx, Phi, filename)
        return
    
    N = len(inv_idx)
    M = len(macro_names)
    
    # 节点：前半是微观，后半是宏观
    node_labels = list(inv_idx.values()) + macro_names
    node_colors = ['#5B9BD5'] * N + ['#ED7D31'] * M
    
    # 微观→宏观流 (Phi)
    sources_micro = []
    targets_macro = []
    values_micro = []
    pi = np.ones(N) / N  # 均匀权重
    for i in range(N):
        sources_micro.append(i)
        targets_macro.append(N + labels[i])
        values_micro.append(1.0)  # 每个微观态权重=1
    
    # 宏观→宏观流 (P_macro)
    sources_macro = []
    targets_macro = []
    values_macro = []
    for i in range(M):
        for j in range(M):
            if P_macro[i, j] > 0.05:  # 只画 >5% 的流
                sources_macro.append(N + i)
                targets_macro.append(N + j)
                values_macro.append(P_macro[i, j] * 10)  # 放大便于可视化
    
    fig = Figure(data=[Sankey(
        node=dict(label=node_labels, color=node_colors, pad=15, thickness=15),
        link=dict(
            source=sources_micro + sources_macro,
            target=targets_macro + targets_macro,
            value=values_micro + values_macro,
            color=['rgba(91,155,213,0.3)'] * len(sources_micro) +
                  ['rgba(237,125,49,0.5)'] * len(sources_macro)
        )
    )])
    fig.update_layout(title_text="《道德经》概念粗粒化桑基图", font_size=12, width=1000, height=700)
    pio.write_image(fig, os.path.join(OUTPUT_DIR, filename), scale=2)
    pio.write_html(fig, os.path.join(OUTPUT_DIR, 'sankey_interactive.html'))
    print(f"  ✓ 桑基图已保存: {filename} + sankey_interactive.html")


def plot_sankey_matplotlib(P_macro, labels, macro_names, idx, inv_idx, Phi, filename):
    """Matplotlib 版桑基图替代方案"""
    fig, ax = plt.subplots(figsize=(10, 8))
    N = len(inv_idx)
    M = len(macro_names)
    
    # 左侧微观态位置
    left_y = np.linspace(0.9, 0.1, N)
    # 右侧宏观态位置
    right_y = np.linspace(0.8, 0.2, M)
    
    # 画微观节点
    for i, name in inv_idx.items():
        ax.plot(0.1, left_y[i], 'o', color='#5B9BD5', markersize=8)
        ax.text(0.08, left_y[i], name, ha='right', va='center', fontsize=9)
    
    # 画宏观节点
    for j, name in enumerate(macro_names):
        ax.plot(0.9, right_y[j], 's', color='#ED7D31', markersize=12)
        ax.text(0.92, right_y[j], name, ha='left', va='center', fontsize=11, fontweight='bold')
    
    # 画流带（简化版）
    for i in range(N):
        j = labels[i]
        ax.plot([0.1, 0.9], [left_y[i], right_y[j]],
                '-', color='#5B9BD5', alpha=0.15, linewidth=1.5)
    
    # 画宏观转移
    for i in range(M):
        for j in range(M):
            if P_macro[i, j] > 0.1:
                ax.annotate('', xy=(0.92, right_y[j]), xytext=(0.88, right_y[i]),
                           arrowprops=dict(arrowstyle='->', color='#ED7D31',
                                           lw=P_macro[i,j]*3, alpha=0.7))
    
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.axis('off')
    ax.set_title('《道德经》概念粗粒化流图（简化版）', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()


def plot_theme_river(chapter_seqs, labels_map, macro_names, filename='theme_river.png'):
    """主题河流图：每章中各宏观态的概念密度"""
    chapters = sorted(chapter_seqs.keys())
    M = len(macro_names)
    
    # 构建每章的宏观态计数
    chapter_macro = {}
    for ch in chapters:
        seq = chapter_seqs[ch]
        counts = np.zeros(M)
        for concept in seq:
            if concept in labels_map:
                counts[labels_map[concept]] += 1
        total = counts.sum()
        if total > 0:
            counts = counts / total
        chapter_macro[ch] = counts
    
    # 堆叠面积图
    fig, ax = plt.subplots(figsize=(18, 7))
    data = np.array([chapter_macro[ch] for ch in chapters])
    x = np.arange(len(chapters))
    
    colors = plt.cm.Set2(np.linspace(0, 1, M))
    ax.stackplot(x, [data[:, j] for j in range(M)],
                 labels=macro_names, colors=colors, alpha=0.85)
    
    ax.set_xticks(x[::5])
    ax.set_xticklabels([f'第{chapters[i]}章' for i in range(0, len(chapters), 5)],
                        rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('概念密度', fontsize=12)
    ax.set_title('《道德经》主题河流图（宏观义理在81章中的兴衰）',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.set_xlim(0, len(chapters)-1)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()


def plot_ei_curve(seq, max_M=20, filename='ei_curve.png'):
    """因果涌现曲线：不同宏观状态数下的归一化 EI"""
    P_micro, C, idx, inv_idx = build_transition_matrix(seq, k=1)
    pi_micro = stationary_distribution(P_micro)
    
    M_values = range(2, max_M + 1)
    ei_values = []
    ei_norm_values = []
    
    # 微观基线
    ei_micro = normalized_ei(P_micro, pi_micro)
    
    for M in M_values:
        try:
            labels, _, explained, s, emb = svd_coarse_grain(P_micro, pi_micro, num_macro_states=M)
            P_macro, Phi = build_macro_transition(P_micro, labels, idx, inv_idx)
            pi_macro = stationary_distribution(P_macro)
            ei = normalized_ei(P_macro, pi_macro)
            ei_values.append(effective_information(P_macro, pi_macro))
            ei_norm_values.append(ei)
        except Exception as e:
            ei_values.append(0)
            ei_norm_values.append(0)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：归一化 EI
    ax1 = axes[0]
    ax1.axhline(y=ei_micro, color='red', linestyle='--', linewidth=1.5,
                label=f'微观基线 ({ei_micro:.4f})')
    ax1.plot(list(M_values), ei_norm_values, 'o-', color='#2E75B6', linewidth=2, markersize=6)
    ax1.fill_between(list(M_values), ei_norm_values, alpha=0.1, color='#2E75B6')
    best_M = list(M_values)[np.argmax(ei_norm_values)]
    best_ei = max(ei_norm_values)
    ax1.scatter([best_M], [best_ei], color='red', s=150, zorder=5, marker='*')
    ax1.annotate(f'最优 M={best_M}\nEI={best_ei:.4f}',
                xy=(best_M, best_ei), xytext=(best_M+2, best_ei*0.9),
                fontsize=11, fontweight='bold', color='red',
                arrowprops=dict(arrowstyle='->', color='red'))
    ax1.set_xlabel('宏观状态数 M', fontsize=12)
    ax1.set_ylabel('归一化有效信息 Eff(P_M)', fontsize=12)
    ax1.set_title('因果涌现曲线（归一化 EI vs M）', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 右图：奇异值谱
    ax2 = axes[1]
    F = np.diag(pi_micro) @ P_micro
    _, s_full, _ = np.linalg.svd(F)
    s_norm = s_full / s_full[0]
    ax2.bar(range(1, len(s_norm)+1), s_norm, color='#70AD47', edgecolor='white', linewidth=0.5)
    ax2.set_xlabel('奇异值序号', fontsize=12)
    ax2.set_ylabel('归一化奇异值 σ_i/σ_1', fontsize=12)
    ax2.set_title('稳态流矩阵 F 的奇异值谱', fontsize=13, fontweight='bold')
    # 标注间隙
    gaps = [s_norm[i] - s_norm[i+1] for i in range(len(s_norm)-1)]
    if gaps:
        best_gap = np.argmax(gaps) + 1
        ax2.annotate(f'谱间隙 @K={best_gap}', xy=(best_gap+0.5, s_norm[best_gap]),
                    fontsize=10, fontweight='bold', color='red',
                    arrowprops=dict(arrowstyle='->', color='red'))
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    
    return best_M, best_ei


def plot_concept_network(P, idx, inv_idx, pi, filename='concept_network.png',
                         top_edges=80, min_weight=0.03):
    """微观概念网络图（用 matplotlib 绘制，输出为静态图供 Gephi 导入）"""
    import networkx as nx
    
    N = len(inv_idx)
    G = nx.DiGraph()
    
    # 添加节点
    for i in range(N):
        G.add_node(inv_idx[i], weight=pi[i])
    
    # 添加边（过滤弱边）
    edges = []
    for i in range(N):
        for j in range(N):
            if P[i, j] >= min_weight and i != j:
                edges.append((inv_idx[i], inv_idx[j], P[i, j]))
    
    # 按权重排序，取 top
    edges.sort(key=lambda x: -x[2])
    edges = edges[:top_edges]
    
    for src, dst, w in edges:
        G.add_edge(src, dst, weight=w)
    
    # 布局
    pos = nx.spring_layout(G, k=2.5, iterations=200, seed=42)
    
    fig, ax = plt.subplots(figsize=(16, 14))
    
    # 节点大小 ∝ π
    node_sizes = [pi[idx[n]] * 8000 + 200 for n in G.nodes()]
    # 边宽度 ∝ 权重
    edge_widths = [G[u][v]['weight'] * 8 for u, v in G.edges()]
    edge_colors = [G[u][v]['weight'] for u, v in G.edges()]
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='#5B9BD5',
                           alpha=0.85, edgecolors='#2E75B6', linewidths=1.5, ax=ax)
    edges_drawn = nx.draw_networkx_edges(G, pos, width=edge_widths,
                                         edge_color=edge_colors,
                                         edge_cmap=plt.cm.YlOrRd,
                                         alpha=0.7, arrows=True, arrowsize=12, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, font_family=CN_FONT.get_name(), ax=ax)
    
    # 颜色条
    sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, norm=plt.Normalize(0, max(edge_colors)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, label='转移概率')
    
    ax.set_title('《道德经》微观概念网络图', fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 导出为 GML 供 Gephi 使用
    nx.write_gml(G, os.path.join(OUTPUT_DIR, 'concept_network.gml'))
    print(f"  ✓ 网络图已保存: {filename} + concept_network.gml")


# ============================================================
# 主流程（拆分阶段函数，消除 311 行超长 main）
# 【重构 T12】将单一 main() 拆为多个职责单一的子函数：
#   stage_text_prep     阶段一：文本清洗与概念抽取
#   stage_transition    阶段二：构建转移矩阵
#   stage_stationary    阶段三：平稳分布 + EI
#   stage_coarse_grain  阶段四：SVD 粗粒化
#   stage_lumpability   阶段五：成块性检验
#   run_visualizations  阶段六：6 种可视化
#   stage_k2_compare    阶段七：k=2 对照
#   save_core_outputs   数据保存（消除重复保存逻辑）
#   main                协调器，串联各阶段
# ============================================================

def _print_header(title):
    """打印阶段分隔标题（居中分隔线 + 标题）"""
    print("\n" + "=" * 60)
    print(f"[{title}]")
    print("=" * 60)


def stage_text_prep():
    """阶段一：文本清洗与概念抽取，返回 (full_seq, chapter_seqs)"""
    _print_header("阶段一 文本清洗与概念抽取")
    full_seq, chapter_seqs = build_full_sequence(DAODEJING)
    unique_concepts = sorted(list(set(full_seq)))
    print(f"  ✓ 全书概念序列长度: {len(full_seq)}")
    print(f"  ✓ 唯一概念数 N = {len(unique_concepts)}")
    print(f"  ✓ 概念列表: {unique_concepts}")

    from collections import Counter
    freq = Counter(full_seq)
    print(f"\n  概念频率 TOP 15:")
    for c, n in freq.most_common(15):
        print(f"    {c:>4s}: {n:>3d} 次 ({n/len(full_seq)*100:.1f}%)")

    print(f"\n  每章平均概念数: {np.mean([len(v) for v in chapter_seqs.values()]):.1f}")
    print(f"  总概念观测数: {len(full_seq)}")
    return full_seq, chapter_seqs


def stage_transition(full_seq):
    """阶段二：构建转移矩阵 (k=1)，返回 (P, C, idx, inv_idx, density)"""
    _print_header("阶段二 构建转移矩阵 (k=1)")
    P, C, idx, inv_idx = build_transition_matrix(full_seq, k=1, smoothing=1.0)
    N = P.shape[0]
    print(f"  ✓ 转移矩阵 P: {N}×{N}")

    # 原始计数矩阵密度
    C_nonzero = int((C > 0).sum())
    density = C_nonzero / C.size * 100
    print(f"  ✓ 原始计数矩阵密度: {density:.1f}% (非零元素占比)")
    print(f"  ✓ 非零转移对: {C_nonzero}, 平均每对观测: {C.sum() / max(C_nonzero, 1):.2f} 次")
    return P, C, idx, inv_idx, density


def stage_stationary(P, idx, inv_idx):
    """阶段三：平稳分布 + EI，返回 (pi, ei_raw, ei_norm)"""
    _print_header("阶段三 平稳分布与有效信息")
    pi = stationary_distribution(P)
    print(f"  ✓ 平稳分布 π 已收敛")
    print(f"\n  平稳分布 TOP 10:")
    pi_sorted = sorted(enumerate(pi), key=lambda x: -x[1])
    for i, p in pi_sorted[:10]:
        print(f"    {inv_idx[i]:>4s}: {p:.4f}")

    ei_raw = effective_information(P, pi)
    ei_norm = normalized_ei(P, pi)
    print(f"\n  ✓ EI(P) = {ei_raw:.4f} bits")
    print(f"  ✓ 归一化 EI = {ei_norm:.4f}")
    return pi, ei_raw, ei_norm


def stage_coarse_grain(P, pi, idx, inv_idx, M=6):
    """
    阶段四：SVD 谱分解 + K-Means 粗粒化。
    返回 dict：labels, embedding, s_vals, macro_groups, macro_names,
               P_macro, Phi, pi_macro, ei_macro_norm, explained
    """
    _print_header("阶段四 SVD 谱分解 + K-Means 粗粒化")
    labels, centers, explained, s_vals, embedding = svd_coarse_grain(P, pi, num_macro_states=M)
    print(f"  ✓ SVD 前 {M} 个成分解释方差: {explained*100:.1f}%")
    print(f"  ✓ 奇异值: {[f'{s:.4f}' for s in s_vals[:10]]}")

    N = P.shape[0]
    # 构建标签映射 + 宏观分组
    concept_to_macro = {inv_idx[i]: labels[i] for i in range(N)}
    macro_groups = {}
    for i in range(N):
        m = labels[i]
        macro_groups.setdefault(m, []).append((inv_idx[i], pi[i]))

    print(f"\n  宏观态分组 (M={M}):")
    macro_names = []
    for m in sorted(macro_groups.keys()):
        items = sorted(macro_groups[m], key=lambda x: -x[1])
        name = "+".join([c for c, _ in items[:3]])
        macro_names.append(name)
        print(f"    [{m}] {name}")
        for c, p in items:
            print(f"        {c:>4s} (π={p:.4f})")

    # 构建宏观转移矩阵 + 宏观 EI
    P_macro, Phi = build_macro_transition(P, labels, idx, inv_idx)
    print(f"\n  ✓ 宏观转移矩阵 P_macro: {M}×{M}")
    df_macro = pd.DataFrame(P_macro, index=macro_names, columns=macro_names)
    print(df_macro.round(3).to_string())

    pi_macro = stationary_distribution(P_macro)
    ei_macro_norm = normalized_ei(P_macro, pi_macro)
    print(f"\n  ✓ 宏观 EI = {effective_information(P_macro, pi_macro):.4f} bits")
    print(f"  ✓ 宏观归一化 EI = {ei_macro_norm:.4f}")

    return {
        'labels': labels, 'embedding': embedding, 's_vals': s_vals,
        'concept_to_macro': concept_to_macro, 'macro_groups': macro_groups,
        'macro_names': macro_names, 'P_macro': P_macro, 'Phi': Phi,
        'pi_macro': pi_macro, 'ei_macro_norm': ei_macro_norm,
        'explained': explained,
    }


def stage_lumpability(P, labels, idx, inv_idx, M):
    """阶段五：成块性检验，返回 lump_err"""
    _print_header("阶段五 成块性检验")
    partition = []
    for m in range(M):
        block = [inv_idx[i] for i in range(P.shape[0]) if labels[i] == m]
        partition.append(block)

    lump_err = lumpability_error(P, partition, idx)
    print(f"  ✓ 成块性误差 ε = {lump_err:.6f}")
    if lump_err < 0.01:
        print(f"  ✓ 误差 < 0.01，分组近似成块")
    else:
        print(f"  ⚠ 误差较大，分组不完全成块")
    return lump_err


def save_core_outputs(full_seq, chapter_seqs, P, pi, P_macro, Phi,
                      macro_names, macro_groups, M, ei_norm, ei_macro_norm,
                      lump_err):
    """保存核心数据文件（矩阵 / CSV / 分组文本），返回 df_macro"""
    np.save(os.path.join(OUTPUT_DIR, 'P_matrix.npy'), P)
    np.save(os.path.join(OUTPUT_DIR, 'pi.npy'), pi)
    np.save(os.path.join(OUTPUT_DIR, 'P_macro.npy'), P_macro)
    np.save(os.path.join(OUTPUT_DIR, 'Phi.npy'), Phi)

    # 保存概念序列 CSV（带章节号）
    df_seq = pd.DataFrame([
        {'position': i, 'concept': c, 'chapter': None}
        for i, c in enumerate(full_seq)
    ])
    pos = 0
    for ch in sorted(chapter_seqs.keys()):
        for c in chapter_seqs[ch]:
            df_seq.at[pos, 'chapter'] = ch
            pos += 1
    df_seq.to_csv(os.path.join(OUTPUT_DIR, 'concept_sequence.csv'),
                  index=False, encoding='utf-8-sig')

    # 保存宏观转移矩阵 CSV
    df_macro = pd.DataFrame(P_macro, index=macro_names, columns=macro_names)
    df_macro.to_csv(os.path.join(OUTPUT_DIR, 'P_macro.csv'), encoding='utf-8-sig')

    # 保存分组信息
    with open(os.path.join(OUTPUT_DIR, 'macro_partition.txt'), 'w', encoding='utf-8') as f:
        f.write(f"M = {M}\n")
        f.write(f"微观 EI = {ei_norm:.4f}\n")
        f.write(f"宏观 EI = {ei_macro_norm:.4f}\n")
        f.write(f"成块性误差 = {lump_err:.6f}\n\n")
        for m in sorted(macro_groups.keys()):
            items = sorted(macro_groups[m], key=lambda x: -x[1])
            f.write(f"\n[{m}] {macro_names[m]}\n")
            for c, p in items:
                f.write(f"  {c}: π={p:.4f}\n")

    print(f"  ✓ P_matrix.npy / pi.npy / P_macro.npy / Phi.npy")
    print(f"  ✓ concept_sequence.csv / P_macro.csv / macro_partition.txt")
    return df_macro


def run_visualizations(P, idx, inv_idx, pi, labels, embedding, s_vals,
                       P_macro, macro_names, Phi, chapter_seqs,
                       concept_to_macro, full_seq):
    """阶段六：6 种可视化（每项独立 try-except，单项失败不中断）"""
    _print_header("阶段六 生成 6 种可视化")

    # ① 微观概念网络图
    print("\n  ① 微观概念网络图...")
    try:
        plot_concept_network(P, idx, inv_idx, pi, min_weight=0.04)
    except Exception as e:
        print(f"    ⚠ 失败: {e}")

    # ② 转移矩阵聚类热力图
    print("  ② 转移矩阵聚类热力图...")
    try:
        plot_heatmap(P, idx, "《道德经》微观概念转移矩阵 (k=1)", "heatmap.png")
    except Exception as e:
        print(f"    ⚠ 失败: {e}")

    # ③ 谱空间散点图
    print("  ③ SVD 谱空间散点图...")
    try:
        plot_spectral_scatter(embedding, labels, idx, inv_idx, s_vals)
    except Exception as e:
        print(f"    ⚠ 失败: {e}")

    # ④ 桑基图
    print("  ④ 桑基图（微观→宏观 + 宏观转移）...")
    try:
        plot_sankey(P_macro, labels, macro_names, idx, inv_idx, Phi)
    except Exception as e:
        print(f"    ⚠ 失败（matplotlib 后备已尝试）: {e}")

    # ⑤ 主题河流图
    print("  ⑤ 主题河流图...")
    try:
        plot_theme_river(chapter_seqs, concept_to_macro, macro_names)
    except Exception as e:
        print(f"    ⚠ 失败: {e}")

    # ⑥ 因果涌现曲线
    print("  ⑥ 因果涌现曲线 + 奇异值谱...")
    N = P.shape[0]
    try:
        best_M, best_ei = plot_ei_curve(full_seq, max_M=min(20, N - 1))
        print(f"  ✓ 最优宏观状态数: M = {best_M} (EI = {best_ei:.4f})")
        return best_M
    except Exception as e:
        print(f"    ⚠ 失败: {e}")
        return None


def stage_k2_compare(full_seq, N, density, ei_raw, ei_norm):
    """阶段七：k=2 转移矩阵对照（诊断二阶结构）"""
    _print_header("阶段七 k=2 转移矩阵对照")
    P2_result = build_transition_matrix(full_seq, k=2, smoothing=1.0)
    if len(P2_result) != 6:
        print("  ⚠ k=2 结果格式异常，跳过")
        return

    P2, C2, pair_idx, pair_list, idx2, inv_idx2 = P2_result
    pi2 = stationary_distribution(P2)
    ei2_raw = effective_information(P2, pi2)
    ei2_norm = normalized_ei(P2, pi2)
    n_pairs = P2.shape[0]

    C2_zero = (C2 == 0).sum()
    density2 = (C2.size - C2_zero) / C2.size * 100

    print(f"  ✓ k=2 联合状态数: {n_pairs}")
    print(f"  ✓ 原始计数矩阵密度: {density2:.1f}%")
    print(f"  ✓ EI(P^(2)) = {ei2_raw:.4f} bits")
    print(f"  ✓ 归一化 EI = {ei2_norm:.4f}")

    print(f"\n  ┌─────────────┬──────────┬──────────┬──────────┐")
    print(f"  │    指标     │   k=1    │   k=2    │   差异   │")
    print(f"  ├─────────────┼──────────┼──────────┼──────────┤")
    print(f"  │ 状态数 N    │   {N:>4d}   │   {n_pairs:>4d}   │ {n_pairs-N:>+5d}   │")
    print(f"  │ 密度(%)     │  {density:>6.1f}  │  {density2:>6.1f}  │ {density2-density:>+6.1f}  │")
    print(f"  │ EI(bits)    │ {ei_raw:>8.4f}  │ {ei2_raw:>8.4f}  │ {ei2_raw-ei_raw:>+8.4f}  │")
    print(f"  │ 归一化 EI   │ {ei_norm:>8.4f}  │ {ei2_norm:>8.4f}  │ {ei2_norm-ei_norm:>+8.4f}  │")
    print(f"  └─────────────┴──────────┴──────────┴──────────┘")

    if density2 < 5:
        print(f"\n  ⚠ k=2 密度仅 {density2:.1f}%，数据严重不足")
        print(f"    建议：k=2 结果不可靠，以 k=1 为主分析")


def main():
    """主流程协调器：串联阶段一至七 + 保存 + 总结"""
    print("=" * 70)
    print("  《道德经》马尔科夫链建模 — 完整 Pipeline")
    print("  基于文档2（王弼通行本，含第10章校订）")
    print("=" * 70)

    # ---- 阶段一：文本清洗与概念抽取 ----
    full_seq, chapter_seqs = stage_text_prep()

    # ---- 阶段二：构建转移矩阵 ----
    P, C, idx, inv_idx, density = stage_transition(full_seq)
    N = P.shape[0]

    # ---- 阶段三：平稳分布 + EI ----
    pi, ei_raw, ei_norm = stage_stationary(P, idx, inv_idx)

    # ---- 阶段四：SVD 粗粒化 ----
    cg = stage_coarse_grain(P, pi, idx, inv_idx, M=6)

    # ---- 阶段五：成块性检验 ----
    lump_err = stage_lumpability(P, cg['labels'], idx, inv_idx, M=6)

    # ---- 阶段六：6 种可视化 + 找最优 M ----
    best_M = run_visualizations(
        P, idx, inv_idx, pi, cg['labels'], cg['embedding'], cg['s_vals'],
        cg['P_macro'], cg['macro_names'], cg['Phi'], chapter_seqs,
        cg['concept_to_macro'], full_seq,
    )
    if best_M is None:
        best_M = 6  # 可视化失败时用默认 M=6

    # ---- 保存核心数据 ----
    print("\n" + "=" * 60)
    print("[保存] 矩阵与结果")
    print("=" * 60)
    save_core_outputs(
        full_seq, chapter_seqs, P, pi, cg['P_macro'], cg['Phi'],
        cg['macro_names'], cg['macro_groups'], M=6,
        ei_norm=ei_norm, ei_macro_norm=cg['ei_macro_norm'], lump_err=lump_err,
    )

    # ---- 阶段七：k=2 对照 ----
    stage_k2_compare(full_seq, N, density, ei_raw, ei_norm)

    # ---- 最终总结 ----
    print("\n" + "=" * 70)
    print("  ✓ Pipeline 完成！")
    print(f"  输出目录: {OUTPUT_DIR}/")
    print("=" * 70)
    print(f"\n  核心发现:")
    print(f"    • 微观状态数 N = {N}")
    print(f"    • 概念观测总数 T = {len(full_seq)}")
    print(f"    • 微观归一化 EI = {ei_norm:.4f}")
    print(f"    • 最优宏观状态数 M* = {best_M}")
    print(f"    • 宏观归一化 EI (M=6) = {cg['ei_macro_norm']:.4f}")
    print(f"    • 因果涌现强度 = {cg['ei_macro_norm'] - ei_norm:+.4f}")
    print(f"    • 成块性误差 ε = {lump_err:.6f}")
    print(f"\n  产出文件:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        fsize = os.path.getsize(fpath)
        print(f"    • {f} ({fsize/1024:.1f} KB)")


if __name__ == "__main__":
    main()
