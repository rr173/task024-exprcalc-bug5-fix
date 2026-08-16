// Package expr 实现一个数学表达式计算器：词法分析、递归下降解析与求值。
//
// 支持整数与小数字面量、二元算术(+ - * / % ^)、一元正负号、比较
// (> >= < <= == !=，结果 1/0)、逻辑(&& ||，1/0 且短路)、括号分组、
// 变量绑定与一组内置函数。所有数值为 float64。
//
// 运算符优先级从低到高：|| → && → 比较 → 加减 → 乘除取模 → 一元正负号
// → 幂(^，右结合) → 原子。一元负号优先级低于幂，故 -2^2 = -(2^2) = -4。
//
// 求值过程中的数学错误以可被 errors.Is 判定的哨兵错误返回，不 panic。
// 任何中间或最终结果为 ±Inf 报 ErrOverflow，为 NaN 报 ErrDomain；服务
// 绝不向调用方返回非有限数。
package expr

import (
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"
)

// 求值错误哨兵。调用方可用 errors.Is 判定类别。
var (
	ErrSyntax       = errors.New("表达式语法错误")
	ErrDivideByZero = errors.New("除以零")
	ErrModuloByZero = errors.New("对零取模")
	ErrUndefinedVar = errors.New("未定义的变量")
	ErrUnknownFunc  = errors.New("未知的函数")
	ErrArgCount     = errors.New("函数参数个数错误")
	ErrDomain       = errors.New("数学定义域错误")
	ErrOverflow     = errors.New("数值溢出")
)

// Eval 解析并求值表达式 s，变量值取自 vars。语法错误、求值错误均以 error 返回。
func Eval(s string, vars map[string]float64) (float64, error) {
	if strings.TrimSpace(s) == "" {
		return 0, fmt.Errorf("%w: 表达式为空", ErrSyntax)
	}
	toks, err := tokenize(s)
	if err != nil {
		return 0, err
	}
	p := &parser{toks: toks}
	node, err := p.parseExpr()
	if err != nil {
		return 0, err
	}
	v, err := node.eval(vars)
	if err != nil {
		return 0, err
	}
	// 顶层兜底：保证不返回非有限数。
	return checkFinite(v)
}

// ---------------------------------------------------------------------------
// AST
// ---------------------------------------------------------------------------

type node interface {
	eval(vars map[string]float64) (float64, error)
}

type numberNode struct{ val float64 }

func (n *numberNode) eval(vars map[string]float64) (float64, error) {
	return checkFinite(n.val) // 拒绝 1e400 这类字面量溢出
}

type varNode struct{ name string }

func (n *varNode) eval(vars map[string]float64) (float64, error) {
	v, ok := vars[n.name]
	if !ok {
		return 0, fmt.Errorf("%w: %s", ErrUndefinedVar, n.name)
	}
	return checkFinite(v)
}

type unaryNode struct {
	op   byte // '+' 或 '-'
	x    node
}

func (n *unaryNode) eval(vars map[string]float64) (float64, error) {
	v, err := n.x.eval(vars)
	if err != nil {
		return 0, err
	}
	if n.op == '-' {
		v = -v
	}
	return checkFinite(v)
}

type binaryNode struct {
	op       string
	l, r     node
}

func (n *binaryNode) eval(vars map[string]float64) (float64, error) {
	// 逻辑短路：左操作数决定是否求值右操作数。
	switch n.op {
	case "&&":
		lv, err := n.l.eval(vars)
		if err != nil {
			return 0, err
		}
		if lv == 0 {
			return 0, nil // 短路，不求值右操作数
		}
		rv, err := n.r.eval(vars)
		if err != nil {
			return 0, err
		}
		if rv == 0 {
			return 0, nil
		}
		return 1, nil
	case "||":
		lv, err := n.l.eval(vars)
		if err != nil {
			return 0, err
		}
		if lv != 0 {
			return 1, nil // 短路
		}
		rv, err := n.r.eval(vars)
		if err != nil {
			return 0, err
		}
		if rv != 0 {
			return 1, nil
		}
		return 0, nil
	}

	lv, err := n.l.eval(vars)
	if err != nil {
		return 0, err
	}
	rv, err := n.r.eval(vars)
	if err != nil {
		return 0, err
	}

	switch n.op {
	case "+":
		return checkFinite(lv + rv)
	case "-":
		return checkFinite(lv - rv)
	case "*":
		return checkFinite(lv * rv)
	case "/":
		if rv == 0 {
			return 0, fmt.Errorf("%w: %g / %g", ErrDivideByZero, lv, rv)
		}
		return checkFinite(lv / rv)
	case "%":
		if rv == 0 {
			return 0, fmt.Errorf("%w: %g %% %g", ErrModuloByZero, lv, rv)
		}
		// 截断取模：a - b*trunc(a/b)，结果符号同被除数（与 math.Remainder
		// 的就近偶数取整不同，后者会使 -5 % 3 = 1 而非 -2）。
		return checkFinite(lv - rv*math.Trunc(lv/rv))
	case "^":
		return powImpl(lv, rv)
	case "==":
		return boolToFloat(lv == rv), nil
	case "!=":
		return boolToFloat(lv != rv), nil
	case ">":
		return boolToFloat(lv > rv), nil
	case ">=":
		return boolToFloat(lv >= rv), nil
	case "<":
		return boolToFloat(lv < rv), nil
	case "<=":
		return boolToFloat(lv <= rv), nil
	}
	return 0, fmt.Errorf("%w: 未知运算符 %q", ErrSyntax, n.op)
}

func boolToFloat(b bool) float64 {
	if b {
		return 1
	}
	return 0
}

type callNode struct {
	name string
	args []node
}

func (n *callNode) eval(vars map[string]float64) (float64, error) {
	fn := functions[n.name]
	return fn(n.args, vars)
}

// ---------------------------------------------------------------------------
// 内置函数
// ---------------------------------------------------------------------------

// funcImpl 是一个内置函数的实现：接收未求值的参数节点与变量表，自行决定
// 何时求值（从而支持短路）。返回求值结果或错误。
type funcImpl func(args []node, vars map[string]float64) (float64, error)

var functions = map[string]funcImpl{
	"max":   fnMax,
	"min":   fnMin,
	"abs":   fnUnary("abs", math.Abs),
	"floor": fnUnary("floor", math.Floor),
	"ceil":  fnUnary("ceil", math.Ceil),
	"round": fnUnary("round", math.Round),
	"sqrt":  fnSqrt,
	"pow":   fnPow,
	"clamp": fnClamp,
	"if":    fnIf,
}

// fnUnary 构造要求恰好 1 个参数的单参函数。
func fnUnary(name string, f func(float64) float64) funcImpl {
	return func(args []node, vars map[string]float64) (float64, error) {
		if len(args) != 1 {
			return 0, fmt.Errorf("%w: %s 需要 1 个参数，实际 %d", ErrArgCount, name, len(args))
		}
		v, err := args[0].eval(vars)
		if err != nil {
			return 0, err
		}
		return checkFinite(f(v))
	}
}

func fnMax(args []node, vars map[string]float64) (float64, error) {
	if len(args) < 1 {
		return 0, fmt.Errorf("%w: max 至少需要 1 个参数，实际 0", ErrArgCount)
	}
	best, err := args[0].eval(vars)
	if err != nil {
		return 0, err
	}
	for _, a := range args[1:] {
		v, err := a.eval(vars)
		if err != nil {
			return 0, err
		}
		if v > best {
			best = v
		}
	}
	return best, nil
}

func fnMin(args []node, vars map[string]float64) (float64, error) {
	if len(args) < 1 {
		return 0, fmt.Errorf("%w: min 至少需要 1 个参数，实际 0", ErrArgCount)
	}
	best, err := args[0].eval(vars)
	if err != nil {
		return 0, err
	}
	for _, a := range args[1:] {
		v, err := a.eval(vars)
		if err != nil {
			return 0, err
		}
		if v > best {
			best = v
		}
	}
	return best, nil
}

func fnSqrt(args []node, vars map[string]float64) (float64, error) {
	if len(args) != 1 {
		return 0, fmt.Errorf("%w: sqrt 需要 1 个参数，实际 %d", ErrArgCount, len(args))
	}
	v, err := args[0].eval(vars)
	if err != nil {
		return 0, err
	}
	if v < 0 {
		return 0, fmt.Errorf("%w: sqrt 对负数 %g 无定义", ErrDomain, v)
	}
	return checkFinite(math.Sqrt(v))
}

func fnPow(args []node, vars map[string]float64) (float64, error) {
	if len(args) != 2 {
		return 0, fmt.Errorf("%w: pow 需要 2 个参数，实际 %d", ErrArgCount, len(args))
	}
	b, err := args[0].eval(vars)
	if err != nil {
		return 0, err
	}
	e, err := args[1].eval(vars)
	if err != nil {
		return 0, err
	}
	return powImpl(b, e)
}

func fnClamp(args []node, vars map[string]float64) (float64, error) {
	if len(args) != 3 {
		return 0, fmt.Errorf("%w: clamp 需要 3 个参数，实际 %d", ErrArgCount, len(args))
	}
	x, err := args[0].eval(vars)
	if err != nil {
		return 0, err
	}
	lo, err := args[1].eval(vars)
	if err != nil {
		return 0, err
	}
	hi, err := args[2].eval(vars)
	if err != nil {
		return 0, err
	}
	if lo > hi {
		return 0, fmt.Errorf("%w: clamp 下界 %g 大于上界 %g", ErrDomain, lo, hi)
	}
	if x < lo {
		return lo, nil
	}
	if x > hi {
		return hi, nil
	}
	return x, nil
}

// fnIf 实现 if(cond,a,b)：cond 非零返回 a，否则返回 b；未选中分支不求值。
func fnIf(args []node, vars map[string]float64) (float64, error) {
	if len(args) != 3 {
		return 0, fmt.Errorf("%w: if 需要 3 个参数，实际 %d", ErrArgCount, len(args))
	}
	cond, err := args[0].eval(vars)
	if err != nil {
		return 0, err
	}
	if cond != 0 {
		return args[1].eval(vars)
	}
	return args[2].eval(vars)
}

// powImpl 实现 ^ 运算符与 pow 函数的共享语义。
//   - 0 ^ 0 = 1（约定）；0 的正数次方 = 0；0 的负数次方 = 除零错。
//   - 负底数 + 非整数指数 = 定义域错（不产生复数）。
//   - 其余交由 math.Pow，并检查结果有限性（溢出→ErrOverflow）。
func powImpl(base, exp float64) (float64, error) {
	if base == 0 {
		if exp == 0 {
			return 0, nil
		}
		if exp < 0 {
			return 0, fmt.Errorf("%w: 0 的 %g 次方", ErrDivideByZero, exp)
		}
		return 0, nil // exp > 0
	}
	if base < 0 && exp != math.Trunc(exp) {
		return 0, fmt.Errorf("%w: 负底数 %g 的非整数次方 %g", ErrDomain, base, exp)
	}
	return checkFinite(math.Pow(base, exp))
}

// checkFinite 保证结果有限：±Inf→ErrOverflow，NaN→ErrDomain。
func checkFinite(v float64) (float64, error) {
	if math.IsNaN(v) {
		return 0, ErrDomain
	}
	if math.IsInf(v, 0) {
		return 0, ErrOverflow
	}
	return v, nil
}

// ---------------------------------------------------------------------------
// 词法分析
// ---------------------------------------------------------------------------

type tokKind int

const (
	tokEOF tokKind = iota
	tokNumber
	tokIdent
	tokOp
)

type token struct {
	kind  tokKind
	num   float64
	ident string
	op    string
	pos   int
}

func (t token) text() string {
	switch t.kind {
	case tokNumber:
		return strconv.FormatFloat(t.num, 'g', -1, 64)
	case tokIdent:
		return t.ident
	case tokOp:
		return t.op
	}
	return "<结束>"
}

func isIdentStart(c byte) bool {
	return c == '_' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
}

func isIdentPart(c byte) bool {
	return isIdentStart(c) || (c >= '0' && c <= '9')
}

func isDigit(c byte) bool { return c >= '0' && c <= '9' }

func tokenize(s string) ([]token, error) {
	var toks []token
	i := 0
	for i < len(s) {
		c := s[i]
		switch {
		case c == ' ' || c == '\t' || c == '\n' || c == '\r':
			i++
			continue
		case isDigit(c):
			start := i
			for i < len(s) && isDigit(s[i]) {
				i++
			}
			if i < len(s) && s[i] == '.' {
				i++
				for i < len(s) && isDigit(s[i]) {
					i++
				}
			}
			if i < len(s) && (s[i] == 'e' || s[i] == 'E') {
				i++
				if i < len(s) && (s[i] == '+' || s[i] == '-') {
					i++
				}
				for i < len(s) && isDigit(s[i]) {
					i++
				}
			}
			num, err := strconv.ParseFloat(s[start:i], 64)
			if err != nil {
				return nil, fmt.Errorf("%w: 非法数字 %q", ErrSyntax, s[start:i])
			}
			toks = append(toks, token{kind: tokNumber, num: num, pos: start})
		case isIdentStart(c):
			start := i
			for i < len(s) && isIdentPart(s[i]) {
				i++
			}
			toks = append(toks, token{kind: tokIdent, ident: s[start:i], pos: start})
		default:
			if op, n, ok := matchOp(s, i); ok {
				toks = append(toks, token{kind: tokOp, op: op, pos: i})
				i += n
			} else {
				return nil, fmt.Errorf("%w: 非法字符 %q 于位置 %d", ErrSyntax, string(c), i)
			}
		}
	}
	toks = append(toks, token{kind: tokEOF, pos: len(s)})
	return toks, nil
}

// matchOp 在 s[pos:] 匹配运算符，优先匹配两字符。返回 (op, 长度, 是否命中)。
func matchOp(s string, pos int) (string, int, bool) {
	if pos+1 < len(s) {
		switch s[pos : pos+2] {
		case "&&", "||", "==", "!=", ">=", "<=":
			return s[pos : pos+2], 2, true
		}
	}
	if pos < len(s) {
		switch s[pos] {
		case '+', '-', '*', '/', '%', '^', '(', ')', ',', '>', '<':
			return string(s[pos]), 1, true
		}
	}
	return "", 0, false
}

// ---------------------------------------------------------------------------
// 递归下降解析
// ---------------------------------------------------------------------------

type parser struct {
	toks []token
	pos  int
}

func (p *parser) peek() token { return p.toks[p.pos] }

func (p *parser) next() token {
	t := p.toks[p.pos]
	p.pos++
	return t
}

func (p *parser) parseExpr() (node, error) {
	n, err := p.parseOr()
	if err != nil {
		return nil, err
	}
	if p.peek().kind != tokEOF {
		return nil, fmt.Errorf("%w: 多余的记号 %q", ErrSyntax, p.peek().text())
	}
	return n, nil
}

func (p *parser) parseOr() (node, error) {
	left, err := p.parseAnd()
	if err != nil {
		return nil, err
	}
	for p.peek().kind == tokOp && p.peek().op == "||" {
		p.next()
		right, err := p.parseAnd()
		if err != nil {
			return nil, err
		}
		left = &binaryNode{op: "||", l: left, r: right}
	}
	return left, nil
}

func (p *parser) parseAnd() (node, error) {
	left, err := p.parseEquality()
	if err != nil {
		return nil, err
	}
	for p.peek().kind == tokOp && p.peek().op == "&&" {
		p.next()
		right, err := p.parseEquality()
		if err != nil {
			return nil, err
		}
		left = &binaryNode{op: "&&", l: left, r: right}
	}
	return left, nil
}

func (p *parser) parseEquality() (node, error) {
	left, err := p.parseComparison()
	if err != nil {
		return nil, err
	}
	for p.peek().kind == tokOp && (p.peek().op == "==" || p.peek().op == "!=") {
		op := p.next().op
		right, err := p.parseComparison()
		if err != nil {
			return nil, err
		}
		left = &binaryNode{op: op, l: left, r: right}
	}
	return left, nil
}

func (p *parser) parseComparison() (node, error) {
	left, err := p.parseAdd()
	if err != nil {
		return nil, err
	}
	for p.peek().kind == tokOp && (p.peek().op == ">" || p.peek().op == ">=" || p.peek().op == "<" || p.peek().op == "<=") {
		op := p.next().op
		right, err := p.parseAdd()
		if err != nil {
			return nil, err
		}
		left = &binaryNode{op: op, l: left, r: right}
	}
	return left, nil
}

func (p *parser) parseAdd() (node, error) {
	left, err := p.parseMul()
	if err != nil {
		return nil, err
	}
	for p.peek().kind == tokOp && (p.peek().op == "+" || p.peek().op == "-") {
		op := p.next().op
		right, err := p.parseMul()
		if err != nil {
			return nil, err
		}
		left = &binaryNode{op: op, l: left, r: right}
	}
	return left, nil
}

func (p *parser) parseMul() (node, error) {
	left, err := p.parseUnary()
	if err != nil {
		return nil, err
	}
	for p.peek().kind == tokOp && (p.peek().op == "*" || p.peek().op == "/" || p.peek().op == "%") {
		op := p.next().op
		right, err := p.parseUnary()
		if err != nil {
			return nil, err
		}
		left = &binaryNode{op: op, l: left, r: right}
	}
	return left, nil
}

// parseUnary 处理一元正负号；一元优先级低于幂，故 -2^2 = -(2^2) = -4。
func (p *parser) parseUnary() (node, error) {
	if p.peek().kind == tokOp && (p.peek().op == "-" || p.peek().op == "+") {
		op := p.next().op[0]
		x, err := p.parseUnary()
		if err != nil {
			return nil, err
		}
		return &unaryNode{op: op, x: x}, nil
	}
	return p.parsePower()
}

// parsePower 解析幂运算，右结合：2^3^2 = 2^(3^2) = 512。
func (p *parser) parsePower() (node, error) {
	base, err := p.parsePrimary()
	if err != nil {
		return nil, err
	}
	if p.peek().kind == tokOp && p.peek().op == "^" {
		p.next()
		exp, err := p.parseUnary() // 右结合，且允许 2^-2
		if err != nil {
			return nil, err
		}
		return &binaryNode{op: "^", l: base, r: exp}, nil
	}
	return base, nil
}

func (p *parser) parsePrimary() (node, error) {
	t := p.peek()
	switch t.kind {
	case tokNumber:
		p.next()
		return &numberNode{val: t.num}, nil
	case tokIdent:
		p.next()
		if p.peek().kind == tokOp && p.peek().op == "(" {
			p.next()
			args, err := p.parseArgs()
			if err != nil {
				return nil, err
			}
			if p.peek().kind != tokOp || p.peek().op != ")" {
				return nil, fmt.Errorf("%w: 函数调用缺少右括号", ErrSyntax)
			}
			p.next()
			return &callNode{name: t.ident, args: args}, nil
		}
		return &varNode{name: t.ident}, nil
	case tokOp:
		if t.op == "(" {
			p.next()
			e, err := p.parseOr()
			if err != nil {
				return nil, err
			}
			if p.peek().kind != tokOp || p.peek().op != ")" {
				return nil, fmt.Errorf("%w: 括号未闭合", ErrSyntax)
			}
			p.next()
			return e, nil
		}
		return nil, fmt.Errorf("%w: 意外的记号 %q", ErrSyntax, t.text())
	case tokEOF:
		return nil, fmt.Errorf("%w: 意外的结束", ErrSyntax)
	}
	return nil, fmt.Errorf("%w: 意外的记号 %q", ErrSyntax, t.text())
}

func (p *parser) parseArgs() ([]node, error) {
	var args []node
	if p.peek().kind == tokOp && p.peek().op == ")" {
		return args, nil // 空参数表，如 foo()
	}
	for {
		a, err := p.parseOr()
		if err != nil {
			return nil, err
		}
		args = append(args, a)
		if p.peek().kind == tokOp && p.peek().op == "," {
			p.next()
			continue
		}
		break
	}
	return args, nil
}
