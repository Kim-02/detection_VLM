#ifndef RISK_ANALYZER_H
#define RISK_ANALYZER_H

#include "yolo_trt.h"
#include <string>
#include <vector>

struct PersonRiskResult {
    Detection person_box;
    bool has_helmet;
    bool has_vest;
    bool is_risk;
    std::string warning_message;
};

class RiskAnalyzer {
public:
    RiskAnalyzer(float contain_ratio_thresh = 0.5f);

    std::vector<PersonRiskResult> analyzePPE(const std::vector<Detection>& dets) const;

    bool hasAnyRisk(const std::vector<PersonRiskResult>& results) const;

    std::string resultsToText(const std::vector<PersonRiskResult>& results) const;

private:
    float intersectionArea(const Detection& a, const Detection& b) const;
    float boxArea(const Detection& b) const;

    bool isInsidePerson(
        const Detection& person,
        const Detection& item
    ) const;

private:
    float contain_ratio_thresh_;
};

#endif