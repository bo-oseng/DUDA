import utils_lib.helper as base_helper
from utils_lib.score import get_cgcd_anchor_score

HYBRID_UPDATE_MODE = "musiq_then_cgcd"
FGRESQ_HYBRID_UPDATE_MODE = "fgresq_then_cgcd"


def normalize_mode_name(mode):
    token = str(mode).strip().lower().replace("-", "").replace("_", "")
    if token in (
        "musiqthencgcd",
        "musiqthencgcdscore",
        "musiqcgcd",
        "musiqcgcdscore",
        "musiqcgcdfallback",
        "musiqthencgcdanchor",
    ):
        return HYBRID_UPDATE_MODE
    if token in (
        "fgresqthencgcd",
        "fgresqthencgcdscore",
        "fgresqcgcd",
        "fgresqcgcdscore",
        "fgresqcgcdfallback",
        "fgresqthencgcdanchor",
    ):
        return FGRESQ_HYBRID_UPDATE_MODE
    return base_helper.normalize_mode_name(mode)


def is_hybrid_mode(mode):
    return normalize_mode_name(mode) in {HYBRID_UPDATE_MODE, FGRESQ_HYBRID_UPDATE_MODE}


def _select_score_backend(pseudo_update_mode, epoch, dino_model, cgcd_model):
    pseudo_update_mode = normalize_mode_name(pseudo_update_mode)
    if pseudo_update_mode == "musiq":
        return True, False
    if pseudo_update_mode == "fgresq":
        return True, True
    if pseudo_update_mode == "cgcd":
        return False, False
    if pseudo_update_mode == "alternate":
        use_musiq = (epoch % 2 == 0) or (dino_model is None) or (cgcd_model is None)
        return use_musiq, False
    if pseudo_update_mode in {HYBRID_UPDATE_MODE, FGRESQ_HYBRID_UPDATE_MODE}:
        raise ValueError(
            f"{pseudo_update_mode} should be handled by the caller because it needs two score passes."
        )
    raise ValueError(f"Unsupported pseudo_update_mode: {pseudo_update_mode}")



def _compute_musiq_family_scores(iqa_metric, teacher_predict, student_predict, score_reference, pseudo_list, use_fgresq):
    if use_fgresq:
        score_teacher = base_helper.get_fgresq_score(iqa_metric, teacher_predict)
        score_student = base_helper.get_fgresq_score(iqa_metric, student_predict)
        if score_reference is None:
            score_reference = base_helper.get_fgresq_score(iqa_metric, pseudo_list)
        return score_teacher, score_student, score_reference, "FGResQ"

    score_teacher = base_helper.get_musiq_score(iqa_metric, teacher_predict)
    score_student = base_helper.get_musiq_score(iqa_metric, student_predict)
    if score_reference is None:
        score_reference = base_helper.get_musiq_score(iqa_metric, pseudo_list)
    return score_teacher, score_student, score_reference, "MUSIQ"



def _compute_cgcd_anchor_scores(
    teacher_predict,
    student_predict,
    pseudo_list,
    dino_model,
    cgcd_model,
    dino_transform,
    orig2classifier,
    amp_dtype,
    clear_weight,
    neg_weight,
    softmin_tau,
):
    if dino_model is None or cgcd_model is None or dino_transform is None:
        raise ValueError("CGCD anchor scoring requires dino_model, cgcd_model, and dino_transform.")

    clear_idx = orig2classifier[0] if orig2classifier is not None else 0
    score_teacher = get_cgcd_anchor_score(
        teacher_predict,
        dino_model,
        cgcd_model,
        dino_transform,
        clear_idx,
        amp_dtype=amp_dtype,
        clear_weight=clear_weight,
        neg_weight=neg_weight,
        softmin_tau=softmin_tau,
    )
    score_student = get_cgcd_anchor_score(
        student_predict,
        dino_model,
        cgcd_model,
        dino_transform,
        clear_idx,
        amp_dtype=amp_dtype,
        clear_weight=clear_weight,
        neg_weight=neg_weight,
        softmin_tau=softmin_tau,
    )
    score_reference = get_cgcd_anchor_score(
        pseudo_list,
        dino_model,
        cgcd_model,
        dino_transform,
        clear_idx,
        amp_dtype=amp_dtype,
        clear_weight=clear_weight,
        neg_weight=neg_weight,
        softmin_tau=softmin_tau,
    )
    return score_teacher, score_student, score_reference, "CGCD_anchor"



def compute_reliable_score_triplet(
    iqa_metric,
    teacher_predict,
    student_predict,
    score_reference,
    pseudo_list,
    pseudo_update_mode="musiq",
    dino_model=None,
    cgcd_model=None,
    dino_transform=None,
    epoch=0,
    orig2classifier=None,
    amp_dtype=None,
    clear_weight=1.0,
    neg_weight=1.0,
    softmin_tau=0.0,
):
    use_musiq, use_fgresq = _select_score_backend(pseudo_update_mode, epoch, dino_model, cgcd_model)
    if use_musiq:
        return _compute_musiq_family_scores(
            iqa_metric=iqa_metric,
            teacher_predict=teacher_predict,
            student_predict=student_predict,
            score_reference=score_reference,
            pseudo_list=pseudo_list,
            use_fgresq=use_fgresq,
        )

    return _compute_cgcd_anchor_scores(
        teacher_predict=teacher_predict,
        student_predict=student_predict,
        pseudo_list=pseudo_list,
        dino_model=dino_model,
        cgcd_model=cgcd_model,
        dino_transform=dino_transform,
        orig2classifier=orig2classifier,
        amp_dtype=amp_dtype,
        clear_weight=clear_weight,
        neg_weight=neg_weight,
        softmin_tau=softmin_tau,
    )
