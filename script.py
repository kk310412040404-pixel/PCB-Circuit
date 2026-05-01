from collections import namedtuple
import rrc_evaluation_funcs
import importlib
import numpy as np
from shapely.geometry import Polygon as plg

def evaluation_imports():
    return {'shapely.geometry':'plg', 'numpy':'np'}

def default_evaluation_params():
    return {
            'IOU_CONSTRAINT' :  0.33,
            'AREA_PRECISION_CONSTRAINT' :0.5,
            'WORD_SPOTTING' :False,
            'MIN_LENGTH_CARE_WORD' :1,
            'GT_SAMPLE_NAME_2_ID':  '([0-9a-zA-Z_]+).txt',
            'DET_SAMPLE_NAME_2_ID':  '([0-9a-zA-Z_]+).txt',
            'LTRB':False,
            'CRLF':False,
            'CONFIDENCES':True,
            'SPECIAL_CHARACTERS': '!?.:,*"()·[]/\'',
            'ONLY_REMOVE_FIRST_LAST_CHARACTER' : True
        }

def validate_data(gtFilePath, submFilePath, evaluationParams):
    gt = rrc_evaluation_funcs.load_zip_file(gtFilePath, evaluationParams['GT_SAMPLE_NAME_2_ID'])
    subm = rrc_evaluation_funcs.load_zip_file(submFilePath, evaluationParams['DET_SAMPLE_NAME_2_ID'], True)
    for k in gt:
        rrc_evaluation_funcs.validate_lines_in_file(k,gt[k],evaluationParams['CRLF'],evaluationParams['LTRB'],True)
    for k in subm:
        if (k in gt) == False :
            raise Exception("The sample %s not present in GT" %k)
        rrc_evaluation_funcs.validate_lines_in_file(k,subm[k],evaluationParams['CRLF'],evaluationParams['LTRB'],True,evaluationParams['CONFIDENCES'])

def evaluate_method(gtFilePath, submFilePath, evaluationParams):
    def polygon_from_points(points,correctOffset=False):
        if correctOffset:
            points[2] -= 1; points[4] -= 1; points[5] -= 1; points[7] -= 1
        resBoxes=np.empty([1,8],dtype='int32')
        resBoxes[0,0]=int(points[0]); resBoxes[0,4]=int(points[1])
        resBoxes[0,1]=int(points[2]); resBoxes[0,5]=int(points[3])
        resBoxes[0,2]=int(points[4]); resBoxes[0,6]=int(points[5])
        resBoxes[0,3]=int(points[6]); resBoxes[0,7]=int(points[7])
        pointMat = resBoxes[0].reshape([2,4]).T
        return plg( pointMat)

    def rectangle_to_polygon(rect):
        resBoxes=np.empty([1,8],dtype='int32')
        resBoxes[0,0]=int(rect.xmin); resBoxes[0,4]=int(rect.ymax)
        resBoxes[0,1]=int(rect.xmin); resBoxes[0,5]=int(rect.ymin)
        resBoxes[0,2]=int(rect.xmax); resBoxes[0,6]=int(rect.ymin)
        resBoxes[0,3]=int(rect.xmax); resBoxes[0,7]=int(rect.ymax)
        pointMat = resBoxes[0].reshape([2,4]).T
        return plg( pointMat)

    def get_union(pD,pG):
        return pD.area + pG.area - get_intersection(pD, pG)

    def get_intersection_over_union(pD,pG):
        try:
            return get_intersection(pD, pG) / get_union(pD, pG)
        except:
            return 0

    def get_intersection(pD,pG):
        pInt = pD.intersection(pG)
        if pInt.is_empty: return 0
        return pInt.area

    def compute_ap(confList, matchList,numGtCare):
        correct = 0; AP = 0
        if len(confList)>0:
            confList = np.array(confList)
            matchList = np.array(matchList)
            sorted_ind = np.argsort(-confList)
            confList = confList[sorted_ind]
            matchList = matchList[sorted_ind]
            for n in range(len(confList)):
                match = matchList[n]
                if match:
                    correct += 1
                    AP += float(correct)/(n + 1)
            if numGtCare>0:
                AP /= numGtCare
        return AP

    def transcription_match(transGt,transDet,specialCharacters='!?.:,*"()·[]/\'',onlyRemoveFirstLastCharacterGT=True):
        if onlyRemoveFirstLastCharacterGT:
            if (transGt==transDet): return True
            if specialCharacters.find(transGt[0])>-1 and transGt[1:]==transDet: return True
            if specialCharacters.find(transGt[-1])>-1 and transGt[0:len(transGt)-1]==transDet: return True
            if specialCharacters.find(transGt[0])>-1 and specialCharacters.find(transGt[-1])>-1 and transGt[1:len(transGt)-1]==transDet: return True
            return False
        else:
            while len(transGt)>0 and specialCharacters.find(transGt[0])>-1: transGt = transGt[1:]
            while len(transDet)>0 and specialCharacters.find(transDet[0])>-1: transDet = transDet[1:]
            while len(transGt)>0 and specialCharacters.find(transGt[-1])>-1 : transGt = transGt[0:len(transGt)-1]
            while len(transDet)>0 and specialCharacters.find(transDet[-1])>-1: transDet = transDet[0:len(transDet)-1]
            return transGt == transDet

    perSampleMetrics = {}
    matchedSum = 0
    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
    gt = rrc_evaluation_funcs.load_zip_file(gtFilePath,evaluationParams['GT_SAMPLE_NAME_2_ID'])
    subm = rrc_evaluation_funcs.load_zip_file(submFilePath,evaluationParams['DET_SAMPLE_NAME_2_ID'],True)
    numGlobalCareGt = 0
    numGlobalCareDet = 0
    arrGlobalConfidences = []
    arrGlobalMatches = []

    for resFile in gt:
        gtFile = rrc_evaluation_funcs.decode_utf8(gt[resFile])
        recall = 0; precision = 0; hmean = 0; detCorrect = 0
        gtPols = []; detPols = []; gtTrans = []; detTrans = []
        gtDontCarePolsNum = []; detDontCarePolsNum = []
        detMatchedNums = []; pairs = []
        arrSampleConfidences = []; arrSampleMatch = []
        sampleAP = 0

        pointsList,_,transcriptionsList = rrc_evaluation_funcs.get_tl_line_values_from_file_contents(gtFile,evaluationParams['CRLF'],evaluationParams['LTRB'],True,False)
        for n in range(len(pointsList)):
            points = pointsList[n]
            transcription = transcriptionsList[n]
            dontCare = transcription == "###"
            if evaluationParams['LTRB']:
                gtPol = rectangle_to_polygon(Rectangle(*points))
            else:
                gtPol = polygon_from_points(points)
            gtPols.append(gtPol)
            gtTrans.append(transcription)
            if dontCare:
                gtDontCarePolsNum.append( len(gtPols)-1 )

        if resFile in subm:
            detFile = rrc_evaluation_funcs.decode_utf8(subm[resFile])
            pointsList,confidencesList,transcriptionsList = rrc_evaluation_funcs.get_tl_line_values_from_file_contents(detFile,evaluationParams['CRLF'],evaluationParams['LTRB'],True,evaluationParams['CONFIDENCES'])
            for n in range(len(pointsList)):
                points = pointsList[n]
                transcription = transcriptionsList[n]
                if evaluationParams['LTRB']:
                    detPol = rectangle_to_polygon(Rectangle(*points))
                else:
                    detPol = polygon_from_points(points)
                detPols.append(detPol)
                detTrans.append(transcription)
                if len(gtDontCarePolsNum)>0 :
                    for dontCarePolIdx in gtDontCarePolsNum:
                        dontCarePol = gtPols[dontCarePolIdx]
                        intersected_area = get_intersection(dontCarePol,detPol)
                        pdDimensions = detPol.area
                        precision_temp = 0 if pdDimensions == 0 else intersected_area / pdDimensions
                        if (precision_temp > evaluationParams['AREA_PRECISION_CONSTRAINT'] ):
                            detDontCarePolsNum.append( len(detPols)-1 )
                            break

            if len(gtPols)>0 and len(detPols)>0:
                iouMat = np.empty([len(gtPols),len(detPols)])
                gtRectMat = np.zeros(len(gtPols),np.int8)
                detRectMat = np.zeros(len(detPols),np.int8)
                for gtNum in range(len(gtPols)):
                    for detNum in range(len(detPols)):
                        pG = gtPols[gtNum]; pD = detPols[detNum]
                        iouMat[gtNum,detNum] = get_intersection_over_union(pD,pG)

                for gtNum in range(len(gtPols)):
                    for detNum in range(len(detPols)):
                        if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0 and gtNum not in gtDontCarePolsNum and detNum not in detDontCarePolsNum :
                            if iouMat[gtNum,detNum]>evaluationParams['IOU_CONSTRAINT']:
                                gtRectMat[gtNum] = 1; detRectMat[detNum] = 1
                                correct = transcription_match(gtTrans[gtNum].upper(),detTrans[detNum].upper(),evaluationParams['SPECIAL_CHARACTERS'],evaluationParams['ONLY_REMOVE_FIRST_LAST_CHARACTER'])==True
                                detCorrect += (1 if correct else 0)
                                if correct: detMatchedNums.append(detNum)

            if evaluationParams['CONFIDENCES']:
                for detNum in range(len(detPols)):
                    if detNum not in detDontCarePolsNum :
                        match = detNum in detMatchedNums
                        arrSampleConfidences.append(confidencesList[detNum])
                        arrSampleMatch.append(match)
                        arrGlobalConfidences.append(confidencesList[detNum])
                        arrGlobalMatches.append(match)

        numGtCare = (len(gtPols) - len(gtDontCarePolsNum))
        numDetCare = (len(detPols) - len(detDontCarePolsNum))
        if numGtCare == 0:
            recall = float(1); precision = float(0) if numDetCare >0 else float(1); sampleAP = precision
        else:
            recall = float(detCorrect) / numGtCare
            precision = 0 if numDetCare==0 else float(detCorrect) / numDetCare
            if evaluationParams['CONFIDENCES'] and evaluationParams['WORD_SPOTTING']==False:
                sampleAP = compute_ap(arrSampleConfidences, arrSampleMatch, numGtCare )
        hmean = 0 if (precision + recall)==0 else 2.0 * precision * recall / (precision + recall)
        matchedSum += detCorrect
        numGlobalCareGt += numGtCare
        numGlobalCareDet += numDetCare
        perSampleMetrics[resFile] = {'precision':precision,'recall':recall,'hmean':hmean,'AP':sampleAP}

    AP = 0
    if evaluationParams['CONFIDENCES']:
        AP = compute_ap(arrGlobalConfidences, arrGlobalMatches, numGlobalCareGt)

    methodRecall = 0 if numGlobalCareGt == 0 else float(matchedSum)/numGlobalCareGt
    methodPrecision = 0 if numGlobalCareDet == 0 else float(matchedSum)/numGlobalCareDet
    methodHmean = 0 if methodRecall + methodPrecision==0 else 2* methodRecall * methodPrecision / (methodRecall + methodPrecision)

    methodMetrics = {'precision':methodPrecision, 'recall':methodRecall,'hmean': methodHmean, 'AP': AP  }
    resDict = {'calculated':True,'Message':'','method': methodMetrics,'per_sample': perSampleMetrics}
    return resDict