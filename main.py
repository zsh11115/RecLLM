import time

import Constant
from DataProcess import getUID, readDataset, test_data_item
from Utils import saveProfile, exists_users
from LLMGenerate import LLMGenerate

dataset = 'Books'  # 'Books' 'xxxx' 'xxxx'
subsets = 480  # 480 10 50 200
sampling_method = "SBS"  # 'SBS' 'full' 'recent' 'relevance' 'random' 'centroid_selection' 'boundary_selection_sampling'
parameter = {"distance_threshold": 0.5, "alpha": 1.1, "ratio": 0.6}

if __name__ == '__main__':
    # 读取用户数据
    print("读取用户数据")
    df = readDataset(dataset, subsets)  # 读取数据
    ids = getUID(df)
    mode = Constant.POS_MODE  # 模式
    # mode=Constant.TEST_MODE
    print("对每个用户生成profile")
    # 对每个用户生成profile
    print("length of id:", len(ids))

    # ProfilePath='./Result/{dataset}_{mode}_{subsets}/{sampling_method}/{parameter}/user_profile.json'
    if sampling_method == "SBS":
        parameter_str = '_'.join(map(str, parameter.values()))
    else:
        parameter_str = "none"
    profilePath = Constant.ProfilePath.format(dataset=dataset, subsets=subsets, mode=mode,
                                              sampling_method=sampling_method, parameter=parameter_str)
    user_ids = exists_users(profilePath)
    for id in ids:
        # 如果需要更新user profile 要删除这段代码
        # if id in user_ids:
        #   continue

        candidate_list_test = test_data_item(df, id)  # 测试数据列表
        # time.sleep(30)
        userProfileInfo = LLMGenerate(df, id, mode, sampling_method, parameter)  # 生成用户画像
        print("userProfileInfo:", userProfileInfo)
        input()
        saveProfile(userProfileInfo, profilePath)  # 保存用户画像信息

"""
selected_index [np.int64(0), np.int64(435), np.int64(10), np.int64(421), np.int64(3), np.int64(4), np.int64(28), 
np.int64(361), np.int64(7), np.int64(9), np.int64(40), np.int64(11), np.int64(12), np.int64(13), np.int64(16), 
np.int64(424), np.int64(17), np.int64(18), np.int64(20), np.int64(21), np.int64(37), np.int64(22), np.int64(23), 
np.int64(25), np.int64(394), np.int64(26), np.int64(27), np.int64(174), np.int64(30), np.int64(31), np.int64(33), 
np.int64(217), np.int64(212), np.int64(36), np.int64(43), np.int64(45), np.int64(47), np.int64(419), np.int64(49), 
np.int64(50), np.int64(433), np.int64(51), np.int64(53), np.int64(54), np.int64(58), np.int64(105), np.int64(112), 
np.int64(60), np.int64(64), np.int64(65), np.int64(298), np.int64(385), np.int64(68), np.int64(345), np.int64(259), 
np.int64(71), np.int64(72), np.int64(73), np.int64(74), np.int64(76), np.int64(78), np.int64(80), np.int64(81), 
np.int64(83), np.int64(84), np.int64(85), np.int64(86), np.int64(87), np.int64(88), np.int64(146), np.int64(89), 
np.int64(91), np.int64(92), np.int64(95), np.int64(96), np.int64(97), np.int64(98), np.int64(237), np.int64(100), 
np.int64(103), np.int64(104), np.int64(122), np.int64(107), np.int64(110), np.int64(346), np.int64(111), 
np.int64(114), np.int64(115), np.int64(117), np.int64(118), np.int64(120), np.int64(123), np.int64(124), 
np.int64(127), np.int64(130), np.int64(132), np.int64(133), np.int64(134), np.int64(136), np.int64(139), 
np.int64(140), np.int64(141), np.int64(142), np.int64(182), np.int64(144), np.int64(145), np.int64(147), 
np.int64(148), np.int64(269), np.int64(149), np.int64(150), np.int64(151), np.int64(152), np.int64(160), 
np.int64(153), np.int64(154), np.int64(155), np.int64(157), np.int64(158), np.int64(161), np.int64(162), 
np.int64(177), np.int64(163), np.int64(165), np.int64(167), np.int64(168), np.int64(169), np.int64(171), 
np.int64(172), np.int64(453), np.int64(173), np.int64(175), np.int64(176), np.int64(179), np.int64(180), 
np.int64(184), np.int64(186), np.int64(188), np.int64(189), np.int64(190), np.int64(192), np.int64(193), 
np.int64(194), np.int64(195), np.int64(199), np.int64(201), np.int64(205), np.int64(207), np.int64(210), 
np.int64(211), np.int64(213), np.int64(215), np.int64(219), np.int64(221), np.int64(229), np.int64(233), 
np.int64(223), np.int64(224), np.int64(225), np.int64(230), np.int64(239), np.int64(283), np.int64(267), 
np.int64(240), np.int64(241), np.int64(395), np.int64(243), np.int64(309), np.int64(244), np.int64(245), 
np.int64(246), np.int64(281), np.int64(247), np.int64(248), np.int64(249), np.int64(368), np.int64(250), 
np.int64(415), np.int64(261), np.int64(253), np.int64(459), np.int64(255), np.int64(330), np.int64(256), 
np.int64(257), np.int64(408), np.int64(263), np.int64(264), np.int64(265), np.int64(268), np.int64(272), 
np.int64(273), np.int64(319), np.int64(274), np.int64(275), np.int64(277), np.int64(278), np.int64(279), 
np.int64(282), np.int64(313), np.int64(284), np.int64(285), np.int64(289), np.int64(290), np.int64(291), 
np.int64(338), np.int64(292), np.int64(294), np.int64(295), np.int64(299), np.int64(301), np.int64(303), 
np.int64(464), np.int64(305), np.int64(308), np.int64(328), np.int64(310), np.int64(469), np.int64(312), 
np.int64(315), np.int64(316), np.int64(317), np.int64(318), np.int64(321), np.int64(322), np.int64(323), 
np.int64(326), np.int64(329), np.int64(331), np.int64(332), np.int64(333), np.int64(334), np.int64(335), 
np.int64(337), np.int64(340), np.int64(347), np.int64(348), np.int64(351), np.int64(403), np.int64(369), 
np.int64(354), np.int64(356), np.int64(359), np.int64(364), np.int64(365), np.int64(367), np.int64(366), 
np.int64(374), np.int64(371), np.int64(372), np.int64(373), np.int64(377), np.int64(382), np.int64(383), 
np.int64(384), np.int64(388), np.int64(389), np.int64(396), np.int64(405), np.int64(410), np.int64(411), 
np.int64(412), np.int64(416), np.int64(417), np.int64(420), np.int64(423), np.int64(425), np.int64(426), 
np.int64(428), np.int64(429), np.int64(434), np.int64(438), np.int64(440), np.int64(441), np.int64(445), 
np.int64(446), np.int64(447), np.int64(448), np.int64(449), np.int64(452), np.int64(454), np.int64(456), 
np.int64(460), np.int64(466), np.int64(467), np.int64(470), np.int64(472), np.int64(474), np.int64(478), np.int64(481)]

all selected index [np.int64(97), np.int64(104), np.int64(302), np.int64(37), np.int64(1), np.int64(187), 
np.int64(193), np.int64(344), np.int64(3), np.int64(68), np.int64(89), np.int6 4(431), np.int64(4), np.int64(81), 
np.int64(5), np.int64(143), np.int64(32), np.int64(39), np.int64(6), np.int64(473), np.int64(113), np.int64(441), 
np.int64(8), np.int64(56), np.int64 (9), np.int64(447), np.int64(10), np.int64(12), np.int64(64), np.int64(71), 
np.int64(480), np.int64(87), np.int64(16), np.int64(18), np.int64(19), np.int64(295), np.int64(389), np.int6 4(433), 
np.int64(265), np.int64(23), np.int64(151), np.int64(310), np.int64(24), np.int64(368), np.int64(99), np.int64(27), 
np.int64(309), np.int64(28), np.int64(332), np.int64(140), n p.int64(343), np.int64(454), np.int64(369), 
np.int64(478), np.int64(31), np.int64(235), np.int64(240), np.int64(305), np.int64(264), np.int64(327), np.int64(34), 
np.int64(397), np.int6 4(35), np.int64(38), np.int64(174), np.int64(172), np.int64(44), np.int64(40), np.int64(442), 
np.int64(308), np.int64(41), np.int64(42), np.int64(204), np.int64(228), np.int64(43), np. int64(48), np.int64(77), 
np.int64(220), np.int64(49), np.int64(202), np.int64(361), np.int64(471), np.int64(54), np.int64(448), np.int64(425), 
np.int64(142), np.int64(57), np.int64(460 ), np.int64(58), np.int64(328), np.int64(59), np.int64(61), np.int64(258), 
np.int64(195), np.int64(365), np.int64(439), np.int64(63), np.int64(66), np.int64(67), np.int64(379), np.int6 4(250), 
np.int64(95), np.int64(352), np.int64(275), np.int64(256), np.int64(74), np.int64(362), np.int64(76), np.int64(427), 
np.int64(409), np.int64(79), np.int64(476), np.int64(175), np.int64(80), np.int64(375), np.int64(83), np.int64(125), 
np.int64(117), np.int64(216), np.int64(86), np.int64(90), np.int64(114), np.int64(215), np.int64(92), np.int64(423), 
np.int64( 188), np.int64(252), np.int64(94), np.int64(98), np.int64(115), np.int64(100), np.int64(102), 
np.int64(121), np.int64(396), np.int64(323), np.int64(116), np.int64(107), np.int64(108), np.int64(451), 
np.int64(209), np.int64(363), np.int64(162), np.int64(167), np.int64(127), np.int64(165), np.int64(122), 
np.int64(381), np.int64(128), np.int64(135), np.int64(136), np.i nt64(137), np.int64(138), np.int64(459), 
np.int64(311), np.int64(168), np.int64(145), np.int64(225), np.int64(292), np.int64(146), np.int64(380), 
np.int64(148), np.int64(445), np.int64 (153), np.int64(329), np.int64(154), np.int64(321), np.int64(449), 
np.int64(183), np.int64(155), np.int64(156), np.int64(160), np.int64(161), np.int64(377), np.int64(164), np.int64(200 
), np.int64(465), np.int64(410), np.int64(247), np.int64(178), np.int64(284), np.int64(182), np.int64(210), 
np.int64(184), np.int64(186), np.int64(190), np.int64(197), np.int64(203), n p.int64(227), np.int64(458), 
np.int64(347), np.int64(205), np.int64(208), np.int64(211), np.int64(213), np.int64(214), np.int64(338), 
np.int64(221), np.int64(217), np.int64(266), np.in t64(226), np.int64(241), np.int64(229), np.int64(337), 
np.int64(232), np.int64(450), np.int64(270), np.int64(273), np.int64(331), np.int64(277), np.int64(467), 
np.int64(279), np.int64( 285), np.int64(288), np.int64(291), np.int64(304), np.int64(316), np.int64(317), 
np.int64(330), np.int64(342), np.int64(346), np.int64(351), np.int64(349), np.int64(370), np.int64(371) , 
np.int64(387), np.int64(390), np.int64(453), np.int64(391), np.int64(416), np.int64(401), np.int64(430), 
np.int64(428), np.int64(404), np.int64(407), np.int64(466), np.int64(418), np .int64(440), np.int64(443), 
np.int64(444), np.int64(483), np.int64(464), np.int64(468), np.int64(470)]

"""


#TODO:对于A124S6MYD45PIQ查与personaX的不同