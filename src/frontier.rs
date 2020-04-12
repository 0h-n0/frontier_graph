use std::cell::RefCell;
use std::rc::Rc;
use std::collections::VecDeque;

#[derive(Debug, Clone, PartialEq)]
pub struct Edge {
    src: usize,
    dst: usize,
}

#[derive(Debug, Clone)]
pub struct Graph {
    number_of_vertices: usize,
    edge_list: Vec<Edge>
}

impl Graph {
    pub fn new(number_of_vertices: usize, edge_list: Vec<Edge>) -> Self {
        Graph {
            number_of_vertices,
            edge_list
        }
    }
    fn get_number_of_vertices(&self) -> usize {
        self.number_of_vertices
    }
    fn get_edge_list(&self) -> &Vec<Edge> {
        &self.edge_list
    }
    fn parse_adj_list_text(self) {
    }
    fn to_string(self) {
    }
}

impl Edge {
    pub fn new(src: usize, dst: usize) -> Self {
        Edge { src, dst }
    }
}

#[derive(Debug, Default, Clone)]
struct TotalId {
    id: RefCell<i64>
}

#[derive(Default, Clone)]
pub struct ZDDNode {
    deg: Option<RefCell<Vec<usize>>>,
    comp: Option<RefCell<Vec<usize>>>,
    indeg: Option<RefCell<Vec<usize>>>,
    outdeg: Option<RefCell<Vec<usize>>>,
    sol: usize,
    zero_child: Option<Rc<RefCell<ZDDNode>>>,
    one_child: Option<Rc<RefCell<ZDDNode>>>,
    id: usize,
}

trait ZDDNodeTrait {
    fn create_root_node(number_of_vertices: usize, id: usize) -> Self;
    fn get_id(&self) -> usize;
    fn set_next_id(&mut self, id: usize);
    fn make_copy(&self, number_of_vertices: usize, id: usize) -> Self;
    fn set_child(&mut self, node: Rc<RefCell<Self>>, child_num: usize);
    fn get_child(&self, child_num: i64) -> Rc<RefCell<Self>>;
}

impl std::fmt::Display for ZDDNode {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let z_id = match self.zero_child.as_ref() {
            Some(v) => v.clone().borrow().id,
            None => 10000,
        };
        let o_id = match self.one_child.as_ref() {
            Some(v) => v.clone().borrow().id,
            None => 10000,
        };
        write!(f, "{}:({}, {})", self.id, z_id, o_id)
    }
}

impl std::fmt::Debug for ZDDNode {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let z_id = match self.zero_child.as_ref() {
            Some(v) => v.clone().borrow().id,
            None => 10000,
        };
        let o_id = match self.one_child.as_ref() {
            Some(v) => v.clone().borrow().id,
            None => 10000,
        };
        write!(f, "{}:(z: {}, o: {})", self.id, z_id, o_id)
    }
}

impl ZDDNodeTrait for ZDDNode {
    fn create_root_node(number_of_vertices: usize, id: usize) -> Self {
        let deg = vec![0; number_of_vertices + 1];
        let mut comp = vec![0; number_of_vertices + 1];
        let indeg = vec![0; number_of_vertices + 1];
        let outdeg = vec![0; number_of_vertices + 1];
        for i in 1..=number_of_vertices {
            comp[i] = i;
        }
        ZDDNode {
            deg: Some(RefCell::new(deg)),
            comp: Some(RefCell::new(comp)),
            indeg: Some(RefCell::new(indeg)),
            outdeg: Some(RefCell::new(outdeg)),
            sol: 0,
            zero_child: None,
            one_child: None,
            id: id,
        }
    }
    fn set_next_id(&mut self, id: usize) {
        self.id = id;
    }
    fn get_id(&self) -> usize {
        self.id
    }
    fn make_copy(&self, number_of_vertices: usize, id: usize) -> Self {
        let mut deg = vec![0; number_of_vertices + 1];
        let mut comp = vec![0; number_of_vertices + 1];
        let mut indeg = vec![0; number_of_vertices + 1];
        let mut outdeg = vec![0; number_of_vertices + 1];
        let self_deg = self.deg.as_ref().unwrap().borrow();
        let self_comp = self.comp.as_ref().unwrap().borrow();
        let self_indeg = self.indeg.as_ref().unwrap().borrow();
        let self_outdeg = self.outdeg.as_ref().unwrap().borrow();

        for i in 1..=number_of_vertices{
            deg[i] = self_deg[i];
            comp[i] = self_comp[i];
            indeg[i] = self_indeg[i];
            outdeg[i] = self_outdeg[i];
        }
        ZDDNode {
            deg: Some(RefCell::new(deg)),
            comp: Some(RefCell::new(comp)),
            indeg: Some(RefCell::new(indeg)),
            outdeg: Some(RefCell::new(outdeg)),
            sol: 0,
            zero_child: None,
            one_child: None,
            id: id,
        }
    }
    fn set_child(&mut self, node: Rc<RefCell<Self>>, child_num: usize) {
        if child_num == 0 {
            self.zero_child = Some(node);
        } else {
            self.one_child = Some(node);
        }
    }
    fn get_child(&self, child_num: i64) -> Rc<RefCell<Self>> {
        if child_num == 0 {
            self.zero_child.as_ref().unwrap().clone()
        } else {
            self.one_child.as_ref().unwrap().clone()
        }
    }
}

pub struct State {
    graph: std::rc::Rc<Graph>,
    s: Vec<usize>,
    t: Vec<usize>,
    frontier: Vec<Vec<usize>>,
}

impl State {
    pub fn new(graph: Graph, start: Vec<usize>, end: Vec<usize>) -> Self {
        let graph = std::rc::Rc::new(graph);
        State {
            s: start,
            t: end,
            graph: graph.clone(),
            frontier: State::compute_frontier(graph.clone()),
        }
    }
    fn compute_frontier(graph: std::rc::Rc<Graph>) -> Vec<Vec<usize>>{
        let edge_list = graph.get_edge_list();
        let mut frontier = vec![vec![]; edge_list.len() + 1];

        for i in 0..edge_list.len() {
            for j in 0..frontier[i].len() {
                let a = frontier[i][j];
                frontier[i + 1].push(a);
            }
            let edge = &edge_list[i];
            let src = edge.src;
            let dst = edge.dst;
            if !frontier[i + 1].contains(&src) {
                frontier[i + 1].push(src)
            }
            if !frontier[i + 1].contains(&dst) {
                frontier[i + 1].push(dst)
            }
            if !State::find_element(graph.clone(), i, src) {
                let mut v = frontier[i + 1].clone().into_iter().filter(|&i| i != src).collect::<Vec<_>>();
                frontier[i + 1].truncate(0);
                frontier[i + 1].append(&mut v);
            }
            if !State::find_element(graph.clone(), i, dst) {
                let mut v = frontier[i + 1].clone().into_iter().filter(|&i| i != dst).collect::<Vec<_>>();
                frontier[i + 1].truncate(0);
                frontier[i + 1].append(&mut v);
            }
        }
        frontier
    }
    fn find_element(graph: std::rc::Rc<Graph>, edge_number: usize, value: usize) -> bool {
        let edge_list = graph.get_edge_list();
        for i in edge_number + 1..edge_list.len() {
            if  value == edge_list[i].src || value == edge_list[i].dst {
                return true
            }
        }
        false
    }
}

#[derive(Debug)]
pub struct ZDD {
    node_list_array: Vec<Vec<Rc<RefCell<ZDDNode>>>>,
}

impl ZDD {
    pub fn get_zeronode() -> ZDDNode {
        let zero_t = ZDDNode {
            deg: None,
            comp: None,
            indeg: None,
            outdeg: None,
            sol: 0,
            zero_child: None,
            one_child: None,
            id: 0,
        };
        zero_t
    }
    pub fn get_onenode() -> ZDDNode {
        let one_t = ZDDNode {
            deg: None,
            comp: None,
            indeg: None,
            outdeg: None,
            sol: 1,
            zero_child: None,
            one_child: None,
            id: 1,
        };
        one_t
    }
    pub fn get_number_of_nodes(&self) -> usize {
        let mut num = 0;
        for i in 1..self.node_list_array.len() {
            num += self.node_list_array[i].len()
        }
        num + 2
    }
    pub fn get_number_of_solutions(&mut self) -> usize {
        let mut i = self.node_list_array.len() - 1;
        let mut max_id = 0;
        while i > 0 {
            for j in 0..self.node_list_array[i].len() {
                let ij_node = self.node_list_array[i][j].clone();
                let lo_node_sol = ij_node.borrow().get_child(0).clone().borrow().sol;
                let hi_node_sol = ij_node.borrow().get_child(1).clone().borrow().sol;
                self.node_list_array[i][j].borrow_mut().sol = lo_node_sol + hi_node_sol;
            }
            i -= 1;
        }
        self.node_list_array[1][0].borrow().sol
    }
    pub fn get_sample(&mut self, idx: usize) -> Result<Vec<usize>, String> {
        let mut i = self.node_list_array.len() - 1;
        let mut max_id = 0;
        while i > 0 {
            for j in 0..self.node_list_array[i].len() {
                let ij_node = self.node_list_array[i][j].clone();
                let lo_node_sol = ij_node.borrow().get_child(0).clone().borrow().sol;
                let hi_node_sol = ij_node.borrow().get_child(1).clone().borrow().sol;
                self.node_list_array[i][j].borrow_mut().sol = lo_node_sol + hi_node_sol;
                if ij_node.borrow().id > max_id {
                    max_id = ij_node.borrow().id;
                }
            }
            i -= 1;
        }
        if idx >= self.node_list_array[1][0].borrow().sol {
            return Err(format!("id[{:?}] > number_solution[{:?}]", idx, self.node_list_array[1][0].borrow().sol));
        }
        let mut i = self.node_list_array.len() - 1;
        let mut solution_array = vec![0; max_id+1];
        let zero_node = Rc::new(RefCell::new(ZDD::get_zeronode()));
        let one_node = Rc::new(RefCell::new(ZDD::get_onenode()));
        let mut node_array: Vec<Rc<RefCell<ZDDNode>>> = vec![zero_node.clone(); max_id+1];
        solution_array[0] = 0;
        solution_array[1] = 1;
        node_array[1] = one_node;
        let mut level_first_array = VecDeque::new();
        while i > 0 {
            for j in 0..self.node_list_array[i].len() {
                let ij_node = self.node_list_array[i][j].clone();
                let lo_node_id = ij_node.borrow().get_child(0).clone().borrow().id;
                let hi_node_id = ij_node.borrow().get_child(1).clone().borrow().id;
                let id = ij_node.borrow().id;
                node_array[id] = ij_node;
                solution_array[id] = solution_array[lo_node_id] + solution_array[hi_node_id];
                if j == 0 {
                    level_first_array.push_front(id);
                }
            }
            i -= 1;
        }
        level_first_array.push_back(8);
        level_first_array.push_back(8);
        let mut current_node = 2;
        let mut result = vec![];
        let mut _idx = idx + 1;
        while (current_node >= 2) {
            let mut is_hi = false;
            let lo_id = node_array[current_node].borrow().get_child(0).clone().borrow().id;
            let hi_id = node_array[current_node].borrow().get_child(1).clone().borrow().id;
            let lo = solution_array[lo_id];
            let hi = solution_array[hi_id];

            if lo == 0 {
                is_hi = true;
            } else if hi == 0 {
                is_hi = false;
            } else {
                if _idx % 2 == 0 {
                    is_hi = true;
                } else {
                    is_hi = false;
                }
                _idx = _idx / 2;
            }

            if is_hi {
                let mut is_break = false;
                for i in 0..level_first_array.len() - 1 {
                    if level_first_array[i] <= current_node &&
                        current_node < level_first_array[i + 1] {
                            result.push(i + 1);
                            is_break = true;
                            break;
                    }
                }
                if !is_break {
                    result.push(level_first_array.len()-2);
                }
                current_node = node_array[current_node].borrow().get_child(1).clone().borrow().id;
            } else {
                current_node = node_array[current_node].borrow().get_child(0).clone().borrow().id;
            }
        }
        Ok(result)
    }
}

pub struct Frontier {
    total_zddnode_id: RefCell<usize>,
    zero_t: ZDDNode,
    one_t: ZDDNode,
}

impl Frontier {
    pub fn new() -> Self{
        let zero_t = ZDDNode {
            deg: None,
            comp: None,
            indeg: None,
            outdeg: None,
            sol: 0,
            zero_child: None,
            one_child: None,
            id: 0,
        };
        let one_t = ZDDNode {
            deg: None,
            comp: None,
            indeg: None,
            outdeg: None,
            sol: 1,
            zero_child: None,
            one_child: None,
            id: 1,
        };
        Self {
            total_zddnode_id: RefCell::new(1),
            zero_t: zero_t,
            one_t: one_t,
        }
    }
    fn get_zddnode_id(&self) -> usize {
        *self.total_zddnode_id.borrow_mut() += 1;
        let next_id = *self.total_zddnode_id.borrow();
        next_id
    }
    pub fn construct(&self, state: &State) -> ZDD {
        let edge_list = state.graph.get_edge_list();
        let mut N = vec![vec![]; edge_list.len() + 2];
        N[1].push(Rc::new(RefCell::new(ZDDNode::create_root_node(state.graph.get_number_of_vertices(),
                                                                    self.get_zddnode_id()))));

        for i in 1..=edge_list.len() {
            let mut n_i_1 = N[i + 1].clone();
            for j in 0..N[i].len() {
                let n_hat = N[i][j].clone();
                for x in 0..=1 {
                    let n_prime = {
                        let ref_n_hat = n_hat.borrow();
                        self.check_terminal(&ref_n_hat, i, x, state)
                    };
                    let n_prime = match n_prime {
                        None => {
                            let mut n_prime = n_hat.borrow().make_copy(state.graph.get_number_of_vertices(),
                                                                       *self.total_zddnode_id.borrow());
                            self.update_info(&n_prime, i, x, state);
                            let n_primeprime = self.find(&n_prime, &n_i_1, i, state);
                            let n_prime = match n_primeprime {
                                Some(v) => v,
                                None => {
                                    n_prime.set_next_id(self.get_zddnode_id());
                                    let new_prime = Rc::new(RefCell::new(n_prime));
                                    n_i_1.push(new_prime.clone());
                                    new_prime
                                }
                            };
                            Some(n_prime)
                        },
                        Some(v) => Some(Rc::new(RefCell::new(v.clone()))),
                    };
                    n_hat.borrow_mut().set_child(n_prime.unwrap(), x);
                }
            }
            N[i + 1] = n_i_1;
        }
        ZDD { node_list_array: N }
    }
    fn check_terminal(&self, n_hat: &ZDDNode,
                      i: usize, x: usize, state: &State) -> Option<&ZDDNode> {
        let edge = &state.graph.get_edge_list()[i - 1];
        let comp = n_hat.comp.as_ref().unwrap().borrow();
        if x == 1 {
            // cycle
            if comp[edge.src] == comp[edge.dst] {
                return Some(&self.zero_t);
            }
        }
        let n_prime = n_hat.make_copy(state.graph.get_number_of_vertices(),
                                          *self.total_zddnode_id.borrow());
        self.update_info(&n_prime, i, x, state);
        let ref_deg = &n_prime.deg.unwrap().into_inner();
        let ref_indeg = &n_prime.indeg.unwrap().into_inner();
        let ref_outdeg = &n_prime.outdeg.unwrap().into_inner();
        for y in 0..=1 {
            let u = match y {
                0 => edge.src,
                _ => edge.dst,
            };
            if state.s.contains(&u) && (ref_indeg[u] > 0 || ref_outdeg[u] > 1 ) {
                return Some(&self.zero_t);
            } else if state.t.contains(&u) && (ref_indeg[u] > 1 || ref_outdeg[u] > 0 ) {
                return Some(&self.zero_t);
            }
            else if (!state.s.contains(&u) && !state.t.contains(&u))
                && (ref_indeg[u] == 0 && ref_outdeg[u] >= 1 ) {
                 return Some(&self.zero_t);
            }
        }
        for y in 0..=1 {
            let u = match y {
                0 => edge.src,
                _ => edge.dst,
            };
            if !state.frontier[i].contains(&u) {
                if (state.s.contains(&u) && ref_outdeg[u] != 1) || (state.t.contains(&u) && ref_indeg[u] != 1) {
                    return Some(&self.zero_t);
                } else if (!state.s.contains(&u) && !state.t.contains(&u)) {
                    if (ref_indeg[u] == 0 && ref_outdeg[u] != 0) {
                        return Some(&self.zero_t);
                    } else if (ref_indeg[u] != 0 && ref_outdeg[u] == 0) {
                        return Some(&self.zero_t);
                    }
                }
            }
        }
        if i == state.graph.edge_list.len() {
            return Some(&self.one_t);
        }
        None
    }

    fn update_info(&self, n_hat: &ZDDNode, i: usize, x: usize, state: &State) {
        let edge = &state.graph.get_edge_list()[i - 1];
        let mut deg = n_hat.deg.as_ref().unwrap().borrow_mut();
        let mut indeg = n_hat.indeg.as_ref().unwrap().borrow_mut();
        let mut outdeg = n_hat.outdeg.as_ref().unwrap().borrow_mut();
        let mut comp = n_hat.comp.as_ref().unwrap().borrow_mut();
        for y in 0..=1 {
            let u = match y {
                0 => edge.src,
                _ => edge.dst,
            };
            if !state.frontier[i - 1].contains(&u) {
                deg[u] = 0;
                comp[u] = u;
            }
        }
        if x == 1 {
            deg[edge.src] += 1;
            deg[edge.dst] += 1;
            outdeg[edge.src] += 1;
            indeg[edge.dst] += 1;
            let (c_max, c_min) = {
                if comp[edge.src] > comp[edge.dst] {
                    (comp[edge.src], comp[edge.dst])
                } else {
                    (comp[edge.dst], comp[edge.src])
                }
            };
            for j in 0..state.frontier[i].len() {
                let u = state.frontier[i][j];
                if comp[u] == c_max {
                    comp[u] = c_min;
                }
            }
        }
    }
    fn find(&self,
            n_prime: &ZDDNode,
            n_i: &Vec<Rc<RefCell<ZDDNode>>>,
            i: usize, state: &State) -> Option<Rc<RefCell<ZDDNode>>> {
        for j in 0..n_i.len() {
            let n_primeprime = n_i[j].clone();
            if self.is_equivalent(&n_prime, n_primeprime.clone(), i, state) {
                return Some(n_primeprime.clone());
            }
        }
        None
    }
    fn is_equivalent(&self,
                     node1: &ZDDNode,
                     node2: Rc<RefCell<ZDDNode>>,
                     i: usize, state: &State) -> bool {
        let frontier = &state.frontier[i];
        let node2 = node2.borrow();
        let n1_deg = node1.deg.as_ref().unwrap().borrow();
        let n1_comp = node1.comp.as_ref().unwrap().borrow();
        let n2_deg = node2.deg.as_ref().unwrap().borrow();
        let n2_comp = node2.comp.as_ref().unwrap().borrow();
        for j in 0..frontier.len() {
            let v = frontier[j];
            if n1_deg[v] != n2_deg[v] {
                return false
            }
            if n1_comp[v] != n2_comp[v] {
                return false
            }
        }
        true
    }
}

pub fn calc_frontier_combination(
    number_of_vertices: usize,
    edge_list: Vec<(usize, usize)>,
    srcs: Vec<usize>,
    dsts: Vec<usize>,
    n_samples: usize) -> Vec<Vec<usize>> {
    let edge_list = edge_list.iter().map(|(s, d)| Edge::new(*s, *d)).collect();
    let g = Graph::new(number_of_vertices, edge_list);
    let state = State::new(g, srcs, dsts);
    let frontier = Frontier::new();
    let mut zdd = frontier.construct(&state);
    let mut out: Vec<Vec<usize>> = vec![];
    let n_sol = zdd.get_number_of_solutions();
    let n_samples = if n_samples > n_sol {
        n_sol
    } else {
        n_samples
    };
    for i in 0..n_samples {
        let ele = zdd.get_sample(i).unwrap();
        out.push(ele);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_calc_frontier_combination() {
        let number_of_vertices: usize = 4;
        let edge_list: Vec<(usize, usize)> = vec![
            (1, 2), (1, 3), (2, 4), (3, 4)];
        let srcs: Vec<usize> = vec![1];
        let dsts: Vec<usize> = vec![4];
        let n_samples: usize = 2;
        let out = calc_frontier_combination(number_of_vertices, edge_list, srcs, dsts, n_samples);
        assert_eq!(out, vec![vec![2, 4], vec![1, 3]]);
    }
}
