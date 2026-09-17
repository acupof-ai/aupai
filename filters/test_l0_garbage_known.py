#!/usr/bin/env python3
"""Shared mutation/known-answer selftest harness for the three Cosmopedia L0 garbage
passes (pass1/pass2/pass3_garbage.py), gen-B 2026-09-17.

Each pass is a list of regexes and one predicate `drops(content)`. This harness enforces the
contract fb set for a hand-curated pattern list:

  1. EVERY listed pattern has at least one real DROP sample that matches it. A pattern whose
     intended garbage no sample can trigger is a DEAD rule (listed but never fires) -- it
     either never matches or was written backwards; the selftest fails naming it.
  2. Each DROP sample matches the SPECIFIC pattern it is paired with, not merely some other
     rule, so the {pattern -> sample} map is an honest per-rule witness (index-aligned).
  3. A set of normal-clean KEEP strings is dropped by NO rule (the false-kill guard).
  4. Mutation: deleting any one pattern from PATTERNS leaks that pattern's witness (drops()
     goes false for it), which is checked in mutation_test_l0_garbage, not here.
"""
import importlib.util
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(modname):
    path = os.path.join(_HERE, f"{modname}.py")
    spec = importlib.util.spec_from_file_location(modname, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# One witness per pattern, INDEX-ALIGNED to each module's PATTERNS. Each string is a short
# realistic phrase that the specific regex was written to catch (a name/title + the trailing
# context the rule requires). If a regex cannot be witnessed, leave None -- the selftest then
# FAILS on it as a suspected dead rule, it never silently skips.
PASS1_DROP = [
    "阳光小学是一所历史悠久的学校",
    "越城区某幼儿园位于城区，创建于一九八零年",
    "青山村中心小学建校历史悠久又名希望小学",
    "本文介绍交通大学（Jiao Tong University）的历史",
    "中南财经政法大学金融学院简介，其前身成立于一九四八年，师资力量雄厚",
    "本专业简介面向高职学生，介绍就业培养课程设置与招生实训",
    "本文讲解如何建立并管理一个研究中心和重点实验室",
    "本文介绍高尔夫教练的挥杆训练技巧与球场练习",
    "截拳道武术大师李小龙的武术精神与格斗功夫",
    "花样体操运动员的传奇人物与教练职业生涯",
    "足球运动员的比赛技巧与规则简介",
    "某位足球运动员的个人简介和职业生涯分析",
    "在体育世界中，“队史第一人”这一概念值得讨论",
    "CTA系列赛是青少年网球选拔的重要赛事，设有单打双打和总决赛",
    "本文分析篮球运动员的体测数据与NBA生涯",
    "钢管舞起源于夜总会，讲究技巧柔韧性，具有成人娱乐文化意义",
    "八极拳与大枪：武术的精髓值得研习",
    "《海贼王》是一部人气动漫，角色剧情丰富",
    "《大帝国》中的虚构人物是一个政治娱乐案例",
    "在《某某》这部作品中，一个令人动容的情感线令人难忘",
    "教程：如何玩《太空冒险》游戏",
    "武林外传中的侠士很多，比如郭巨侠和公孙乌龙",
    "虚拟人物设计：性格、技能与背景设定，也涉及角色设计理论与实践",
    "这是一篇漫画产业发展与《少年》案例分析",
    "游戏塞尔达传说：旷野之息的开放世界设计",
    "虚无械之眼：游戏王中的永续陷阱与策略分析",
    "一款益智网页游戏，玩法规则简单，玩家通过关卡，系统提示猜数字",
    "一款找茬找不同游戏，玩法是找出图片差异，很有教育意义",
    "动作游戏的心理学原理值得分析",
    "矢元小梦是一位偶像艺人，本文介绍其个人资料与家庭背景",
    "追剧过程中的情绪波动与心理变化",
    "如何在生活中找到并保持快乐：丑小羊与疯小羊",
    "《谁人是我生父母》是《新白娘子传奇》中一首经典插曲",
    "烧鸡的起源历史与制作工艺、烹饪做法和文化",
    "这款蛋糕甜品的食材做法与制作口感介绍",
    "日本豆腐的简介、营养价值与由来功效",
    "挂霜地瓜条是一道烹饪工艺做法的红薯甜品",
    "营养学入门：番茄鸡蛋汤的营养价值",
    "本文对某食物的食材与营养成分分析",
    "雪梨煲瘦肉的煲汤做法，介绍营养功效与食材美味",
    "海苔腰果这类零食甜品的制作过程、食材油炸与营养美味",
    "酱笋蒸鱼的做法步骤，需要姜片、葱段、米酒、鲜鱼、糖和盐",
    "作为咖啡师，意式浓缩、卡布奇诺、拿铁的制作配方和咖啡机使用",
    "瓢虫等昆虫的形态特征、生活习性与分布范围和分类",
    "晚花大丁草的分布简介与形态生长环境，双子叶植物纲主要分布于高海拔",
    "大花龙芽草隶属于蔷薇科，具有药用价值和生物学特性与生态分布",
    "本文介绍芋螺的生物特性与生态价值",
    "狭叶女贞种植和养护技术（Ligustrum）要点",
    "某国家森林公园位于山区，面积广阔，海拔较高，旅游资源与景点生态丰富",
    "幸福村位于某省某市某县，地处平原，下辖多个村委会，距县城十公里，东邻甲村西连乙村",
    "济南王府池子、濯缨泉、珍珠泉的历史文化与环境保护和泉群景点",
    "某地位于某省某市某县某镇，介绍其地理",
    "深入探索某省某县某乡的自然美景",
    "乡村发展与旅游结合模式——以某某村为例",
    "学习习近平新时代中国特色社会主义思想的课程理论与实践指导",
    "男科病男性健康与前列腺性功能问题的防治，由名医编著、出版社出版",
    "《方洲新概念》作文丛书是学生提高写作的得力助手和良师益友",
    "《小学语文每周一测》是专为三年级设计的语文练习册",
    "如何选择适合自己的医疗机构——以某某医院为例",
    "某同志的干部晋升路径与履历，最终晋升至高级领导岗位",
    "“优秀士兵”和“优秀士官”的含义介绍",
    "盈地大厦是一个商业综合体广场，位于市中心，是标志性建筑，投资回报率高",
    "喷涂型聚脲涂装工程的技术规范、施工涂层与质量检验防腐",
    "不明飞行物UFO的干扰弹现象与解释",
    "流浪动物救助问题网站，基于Java系统平台开发",
    "如何利用《手册》学习，附《刑法分则案例教程》",
    "连云港大学生电影节活动报道",
    "在社交软件微信上索要红包的心态、心理与动机分析",
    "一部网络小说奇幻修炼题材，有鬼月纵横大陆",
    "清康熙年间的端石砚台、东井砚等文物介绍",
    "修仙文化追求长生不老、得道成仙，涉及道教炼丹与内功修炼",
    "爱新觉罗家族，清代郡王亲王贝子官员的生平事迹与封王仕途俸禄",
    "金大定通宝等古钱铜钱的铸行、形制、钱文与收藏书法货币版别",
    "树根雀梅盆景的盆栽，介绍树根雀梅的栽培、修剪、施肥、浇水、拼接、创作造型上盆",
    "Ctrl+Shift+T 快捷键可以恢复关闭的网页标签页，介绍浏览器重新打开方法",
    "古老的中医药方——犀灰散的基本组成与用途",
    "《医案医话集》由中国中医药出版社出版",
    "这是一个具有深厚历史背景的家族派系",
    "《飞廉的村庄》：诗性散文与乡村情结",
    "一位教育改革者的故事",
    "本文属于流行歌曲衍生内容，也是基于儿歌的泛泛励志教程",
]

PASS2_DROP = [
    "台湾女演员苏晏霈的演艺生涯、参演电视剧与广告作品和粉丝交流群",
    "香山墅园体现了森林景观设计理念，在景观坡地、房地产院落方面",
    "酸石榴饮的成分、功效、制作方法、用法及对慢性支气管炎的作用",
    "龙珠果的种植和栽培，包括选择种植地点、种子采集和处理、播种和种植、田间管理、病虫害防治",
    "训练宠物狗要讲命令训练、行为规范训练与社交训练，注意正面强化和负面强化",
    "如何理解和欣赏《魔王道》漫画，少年热血科幻魔幻题材，漫画之家",
    "椤木石楠是常绿乔木，复伞房花序，介绍形态特征、生活习性与物种价值",
    "手工DIY制作手套娃娃、编织手链、彩绘钥匙扣和手工书制作",
    "某中药制药公司如山西银湖制药，介绍其可靠的中药制药、历史背景、质量认证GMP认证与社会信誉",
    "禅宗圣地大愚寺，马祖建道场，百丈立清规，宜春禅宗文化浓厚",
    "小小彩虹岛是一款2D横版卷轴游戏，游戏职业设计有公会系统、盾卫、魔法师等职业",
    "紫峰大厦ZIFENG位于鼓楼广场，由绿地集团投资，世界第七、中国第四高楼",
    "SKF轴承62208-2RS1是深沟球轴承，滚道与滚动体配合，两面带橡胶密封件",
    "赌石看灰卡、会卡、场口等翡翠原石，皮壳呈灰绿色",
    "某奖学金项目由秦维聪设立，纪念李耳，在鹿邑设立并运营该奖学金",
    "炒田螺配酸笋、紫苏叶，斩螺让田螺吐泥，蒜茸豆豉调味",
    "歌手李圣杰演唱《很想说》，歌词中的情感丰富",
    "成神超市体现武侠文化，武技修神、北冥神功与内功修炼",
    "电影《盗梦空间》由诺兰执导，多层梦境带来独特观影体验与梦境叙事",
    "黑五类食材如黑木耳、黑芝麻、黑米、黑枣，有益肾脏健康，可做黑色食品餐",
    "假日旅游是中学生旅游放松身心、拓宽视野、领略风土人情的好方式",
    "摩托罗拉X301是一款3G手机，介绍手机类型、手机频段和主屏参数",
    "民族娃娃换装游戏可以让孩子了解体验民族服饰，换装游戏含丰富民族服饰",
    "婚礼顾问负责婚礼策划，专业婚礼顾问公司提供服务",
    "鸡蛋面的制作方法、做法1和烹饪技艺",
    "沙市清真寺位于某地，介绍其历史背景与建筑特色",
    "PixelController软件用于像素画创作，培养创意思维",
    "历代瓷壶的鉴赏、收藏与辨伪识真",
    "《冰血暴》是一部电视剧，介绍其剧情、演员与导演",
    "超人与蝙蝠侠是DC宇宙的超级英雄，属于正义联盟",
    "四川香草植物园提供种苗、盆苗、干花、精油等产销单位",
    "云湖乡位于文成县，生态良好，经济发展迅速",
    "课程单元：小城镇发展与规划——以始建镇为例，始建镇位于仁寿县南端",
    "城市居住环境与城市规划设计——以嘉和坊小区为例，嘉和坊小区位于上海市普陀区",
    "如何选择合适的住宅项目：海阔天空·国兴城案例剖析，海阔天空·国兴城",
    "本文是模仿印度风格的音译，属于印度风格音译欧美政要",
    "莫斯提马是《明日方舟》群法干员，伊芙利特的技能很强",
    "如何选择适合校庆演出的曲子，校庆演出曲目推荐",
    "万合天宜的本煜气场十足，是颜值代表",
    "如家连锁酒店（沈阳大北关街店），即如家在沈阳大北关街的门店",
    "轴承品牌LHZ是一家专业生产轴承的企业，30315轴承尺寸参数齐全",
]

PASS3_DROP = [
    "帕尔哈提是中国好声音舞台上的吉他天才",
    "考研政治跟徐涛、腿姐的复习策略",
    "芦笋的营养价值与健康益处",
    "华为OD属于外包岗位，OD岗位招聘说明",
    "麻辣土豆条的做法与营养分析，一道家常美食",
    "综合格斗Mixed Martial Arts即终极格斗，类似UFC赛事",
    "某工作日志软件告而广之，支持管理者查看部门员工",
    "候汉生的履历，涉及军队管理、企业管理与施工生产",
    "刀根之国出自梦幻不思议卷轴",
    "秦腔名家田正武，须生行当拿手戏众多",
    "化滞煎一方出自《医学集成》卷三",
    "选秀节目快乐男声选手钱先勇、江映蓉的快女经历",
    "中医偏方与验方，偏方用生姜、葱白、红糖",
    "阿拉德战记改编自地下城与勇士，开发商NEOPLE",
    "广东省商业企业集团公司，省商业集团的资产重组与经营范围",
    "海事局工作的优缺点，海事局公务员二次分配情况",
    "摄影构图讲究黄金分割、框架构图与构图引导线",
    "爵士上海音乐节，介绍音乐节的爵士风格与历史背景",
    "张士柏高位截瘫却自强不息",
    "重生小说《重生九七》涉及时间旅行与记忆重组、预测彩票号码",
    "现代占星术、星相学用星座预测性格与命运",
    "炒拉条的主料与辅料、面团调制与制作步骤、和面方法",
    "手机摄像头膜、防蓝光膜、淬晶AR膜的贴膜安装与购买建议",
    "中韩电视剧、偶像剧如《来自星星的你》《步步惊心》，编剧导演比较",
    "强光手电筒神火 V8凯瑞兹，关注流明、续航与充电",
    "MapGIS标准分幅栅格地图，GDB企业管理器处理金字塔影像校正",
    "锅炉软化阻垢用阳离子交换树脂再生，解决锅炉结垢，工作压力0.18",
    "葛宇路因中央美术学院记过处分引发路名风波",
    "2012北京榜样评选，公民责任与社会榜样，北京新闻广播主办",
    "明朝末年徽州府一位女性管理家庭，卷入家族争斗的故事",
    "谢霆锋歌曲《如果没有感觉》及Listen Up香港版，谈音乐创作与自我表达",
    "泰山纪念币无梯版，纪念币收藏投资，发行量1.2亿",
    "钟玉良任于都县常务副县长、石城县招商引资等地方自治事务",
    "汗血宝马即阿哈尔捷金马，又称阿哈马",
    "赤城中学获评绿色学校，谈绿色学校的重要性及赤城",
    "张家川村属于交口县石口乡",
    "《英汉双解最新英语新词词典》出版",
    "蒸危爆威是火影忍者中二代目水影的招式",
    "中国商文化博物馆即商丘博物馆新馆",
    "温州市工商局发布市场监管公告",
    "周升勇（网名尐_勇）事迹",
    "如何在网页上复制链接，使用execCommand('Copy')",
    "玛表自然村简介",
    "灌汤鱼肚的做法",
    "原文地址：某网站转载链接",
    "这是一篇读后感3，还收录作文大全",
    "本试卷含填空题（ 单选题（多选（判断（简答（ ）等题型",
    "某课程的教案、课件、导学案、学习任务单、电子课本与复习资料",
    "第一章 总论\n第二章 方法",
]

# Normal text no garbage rule must match (Chinese and English prose + code).
KEEP = [
    "线性回归通过最小二乘法拟合数据分布，并给出可解释的参数估计。",
    "本节讨论二分查找在有序数组上的时间复杂度，其为 O(log n)。",
    "def compute(x):\n    return sum(i * i for i in range(x))\n",
    "The quick brown fox jumps over the lazy dog, a plain English sentence.",
    "实验结果表明，加入正则化后验证集误差显著下降，过拟合得到缓解。",
    "今天天气晴朗，我们在公园散步，讨论了最近读完的一本科普书。",
]

CASES = [
    ("pass1_garbage", PASS1_DROP),
    ("pass2_garbage", PASS2_DROP),
    ("pass3_garbage", PASS3_DROP),
]


def run():
    failures = []
    for modname, witnesses in CASES:
        m = _load(modname)
        pats = m.PATTERNS
        if len(witnesses) != len(pats):
            failures.append(f"{modname}: {len(witnesses)} witnesses for {len(pats)} patterns "
                            "(must be index-aligned one-per-pattern)")
            continue
        for i, (pat, wit) in enumerate(zip(pats, witnesses, strict=True)):
            if wit is None:
                failures.append(f"{modname}[{i}] pattern has NO witness (suspected dead rule): {pat[:50]}")
                continue
            # witness must match ITS paired pattern...
            if not re.search(pat, wit):
                failures.append(f"{modname}[{i}] witness does not match its own pattern:\n"
                                f"  pat={pat[:60]}\n  wit={wit[:60]}")
                continue
            # ...and be dropped by production (it may also match others; that is fine).
            if not m.drops(wit):
                failures.append(f"{modname}[{i}] witness matched its regex but drops() was False: {wit[:40]}")
        # every KEEP string must survive
        for k in KEEP:
            if m.drops(k):
                failures.append(f"{modname}: normal text wrongly dropped: {k[:40]}")
    if failures:
        print("FAIL\n" + "\n".join(failures[:40]))
        print(f"\n{len(failures)} failure(s)")
        return 1
    totals = sum(len(w) for _, w in CASES)
    print(f"l0 garbage selftest ok: {totals} pattern witnesses (one per rule, each matches its "
          f"rule and is dropped), {len(KEEP)} normal keeps survive, across pass1/2/3")
    return 0


if __name__ == "__main__":
    sys.exit(run())
