// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef AdaptiveFixedP_H
#define AdaptiveFixedP_H
#include "lduMatrix.H"
namespace Foam {
class AdaptiveFixedP : public lduMatrix::solver {
public:
    TypeName("AdaptiveFixedP");
    AdaptiveFixedP(const word&, const lduMatrix&,
        const FieldField<Field, scalar>&,const FieldField<Field, scalar>&,
        const lduInterfaceFieldPtrsList&,const dictionary&);
    virtual solverPerformance solve(scalarField&,const scalarField&,
        const direction cmpt=0) const;
    AdaptiveFixedP(const AdaptiveFixedP&)=delete;
    void operator=(const AdaptiveFixedP&)=delete;
};
}
#endif
